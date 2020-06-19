import argparse
import os

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from dataset import Places365
from model import VGGFeature, Generator, Discriminator
from mask import make_mask
import distributed as dist


def requires_grad(module, flag):
    for m in module.parameters():
        m.requires_grad = flag


def d_ls_loss(real_predict, fake_predict):
    loss = (real_predict - 1).pow(2).mean() + fake_predict.pow(2).mean()

    return loss


def g_ls_loss(real_predict, fake_predict):
    loss = (fake_predict - 1).pow(2).mean()

    return loss


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def train(args, dataset, gen, dis, g_ema, device):
    if args.distributed:
        g_module = gen.module
        d_module = dis.module

    else:
        g_module = gen
        d_module = dis

    vgg = VGGFeature("vgg16", [4, 9, 16, 23, 30], use_fc=True).eval().to(device)
    requires_grad(vgg, False)

    g_optim = optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.999))
    d_optim = optim.Adam(dis.parameters(), lr=1e-4, betas=(0, 0.999))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=4,
        sampler=dist.data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    loader_iter = sample_data(loader)

    pbar = range(args.start_iter, args.iter)

    if dist.get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True)

    eps = 1e-8

    for i in pbar:
        real, class_id = next(loader_iter)

        real = real.to(device)
        class_id = class_id.to(device)

        masks = make_mask(real.shape[0], device, args.crop_prob)
        features, fcs = vgg(real)
        features = features + fcs[1:]

        requires_grad(dis, True)
        requires_grad(gen, False)

        real_pred = dis(real, class_id)

        z = torch.randn(args.batch, args.dim_z, device=device)

        fake = gen(z, class_id, features, masks)

        fake_pred = dis(fake, class_id)

        d_loss = d_ls_loss(real_pred, fake_pred)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        z1 = torch.randn(args.batch, args.dim_z, device=device)
        z2 = torch.randn(args.batch, args.dim_z, device=device)

        requires_grad(gen, True)
        requires_grad(dis, False)

        masks = make_mask(real.shape[0], device, args.crop_prob)

        if args.distributed:
            gen.broadcast_buffers = True

        fake1 = gen(z1, class_id, features, masks)

        if args.distributed:
            gen.broadcast_buffers = False

        fake2 = gen(z2, class_id, features, masks)

        fake_pred = dis(fake1, class_id)

        a_loss = g_ls_loss(None, fake_pred)

        features_fake, fcs_fake = vgg(fake1)
        features_fake = features_fake + fcs_fake[1:]

        r_loss = 0

        for f_fake, f_real, m in zip(features_fake, features, masks):
            if f_fake.ndim == 4:
                f_fake = F.max_pool2d(f_fake, 2, ceil_mode=True)
                f_real = F.max_pool2d(f_real, 2, ceil_mode=True)
                f_mask = F.max_pool2d(m, 2, ceil_mode=True)

                r_loss = (
                    r_loss
                    + (F.l1_loss(f_fake, f_real, reduction="none") * f_mask).mean()
                )

            else:
                r_loss = r_loss + F.l1_loss(f_fake, f_real)

        div_z = F.l1_loss(z1, z2, reduction="none").mean(1)
        div_fake = F.l1_loss(fake1, fake2, reduction="none").mean((1, 2, 3))

        d_loss = (div_z / (div_fake + eps)).mean()

        g_loss = a_loss + args.rec_weight * r_loss + args.div_weight * d_loss

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module)

        if dist.get_rank() == 0:
            pbar.set_description(
                f"d: {d_loss.item():.4f}; g: {a_loss.item():.4f}; rec: {r_loss.item():.4f}; div: {d_loss.item():.4f}"
            )

            if i % 100 == 0:
                utils.save_image(
                    fake1,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=int(args.batch ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )

            if i % 10000 == 0:
                torch.save(
                    {
                        "args": args,
                        "g_ema": g_ema.state_dict(),
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--iter", type=int, default=500000)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--dim_class", type=int, default=128)
    parser.add_argument("--rec_weight", type=float, default=0.1)
    parser.add_argument("--div_weight", type=float, default=0.1)
    parser.add_argument("--crop_prob", type=float, default=0.3)
    parser.add_argument("path", metavar="PATH")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset = Places365(args.path, transform=transform)
    args.n_class = dset.n_class

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        dist.synchronize()

    gen = Generator(args.n_class, args.dim_z, args.dim_class).to(device)
    g_ema = Generator(args.n_class, args.dim_z, args.dim_class).to(device)
    accumulate(g_ema, gen, 0)
    dis = Discriminator(args.n_class).to(device)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        gen.load_state_dict(ckpt["g"])
        g_ema.load_state_dict(ckpt["g_ema"])
        dis.load_state_dict(ckpt["d"])

    if args.distributed:
        gen = nn.parallel.DistributedDataParallel(
            gen,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

        dis = nn.parallel.DistributedDataParallel(
            dis,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

    train(args, dset, gen, dis, g_ema, device)
