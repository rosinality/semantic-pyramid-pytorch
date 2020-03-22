import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn


def spectral_norm(module):
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)


def get_activation(name):
    if name == 'leaky_relu':
        activation = nn.LeakyReLU(0.2)

    elif name == 'relu':
        activation = nn.ReLU()

    return activation


class VGGFeature(nn.Module):
    def __init__(self, arch, indices, use_fc=False, normalize=True, min_max=(-1, 1)):
        super().__init__()

        vgg = {
            'vgg16': vgg16,
            'vgg16_bn': vgg16_bn,
            'vgg19': vgg19,
            'vgg19_bn': vgg19_bn,
        }.get(arch)(pretrained=True)

        for p in vgg.parameters():
            p.requires_grad = False

        self.slices = nn.ModuleList()

        for i, j in zip([-1] + indices, indices + [None]):
            if j is None:
                break

            self.slices.append(vgg.features[slice(i + 1, j + 1)])

        self.use_fc = use_fc

        if use_fc:
            self.rest_layer = vgg.features[indices[-1] :]
            self.fc6 = vgg.classifier[:3]
            self.fc7 = vgg.classifier[3:6]
            self.fc8 = vgg.classifier[6]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        val_range = min_max[1] - min_max[0]
        mean = mean * (val_range) + min_max[0]
        std = std * val_range

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.normalize = normalize

    def forward(self, input):
        if self.normalize:
            input = (input - self.mean) / self.std

        features = []

        out = input
        for layer in self.slices:
            out = layer(out)
            features.append(out)

        fcs = []

        if self.use_fc:
            out = self.rest_layer(out)
            out = torch.flatten(F.adaptive_avg_pool2d(out, (7, 7)), 1)

            fc6 = self.fc6(out)
            fc7 = self.fc7(fc6)
            fc8 = self.fc8(fc7)

            fcs = [fc6, fc7, fc8]

        return features, fcs


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, in_channel, embed_dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channel, affine=False)

        self.weight = spectral_norm(nn.Linear(embed_dim, in_channel, bias=False))
        self.bias = spectral_norm(nn.Linear(embed_dim, in_channel, bias=False))

        self.in_channel = in_channel
        self.embed_dim = embed_dim

    def forward(self, input, embed):
        out = self.norm(input)

        batch_size = input.shape[0]

        weight = self.weight(embed).view(batch_size, -1, 1, 1)
        bias = self.bias(embed).view(batch_size, -1, 1, 1)

        out = (weight + 1) * out + bias

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, embed_dim={self.embed_dim})'
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        norm=False,
        embed_dim=None,
        upsample=False,
        downsample=False,
        first=False,
        activation='relu',
    ):
        super().__init__()

        self.first = first
        self.norm = norm

        bias = False if norm else True

        if norm:
            self.norm1 = AdaptiveBatchNorm2d(in_channel, embed_dim)

        if not self.first:
            self.activation1 = get_activation(activation)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

        else:
            self.upsample = None

        self.conv1 = spectral_norm(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=bias)
        )

        if norm:
            self.norm2 = AdaptiveBatchNorm2d(out_channel, embed_dim)

        self.activation2 = get_activation(activation)

        self.conv2 = spectral_norm(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=bias)
        )

        if downsample:
            self.downsample = nn.AvgPool2d(2)

        else:
            self.downsample = None

        self.skip = None

        if in_channel != out_channel or upsample or downsample:
            self.skip = spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))

    def forward(self, input, embed=None):
        out = input

        if self.norm:
            out = self.norm1(out, embed)

        if not self.first:
            out = self.activation1(out)

        if self.upsample:
            out = self.upsample(out)

        out = self.conv1(out)

        if self.norm:
            out = self.norm2(out, embed)

        out = self.activation2(out)
        out = self.conv2(out)

        if self.downsample:
            out = self.downsample(out)

        skip = input

        if self.skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            if self.downsample and self.first:
                skip = self.downsample(skip)

            skip = self.skip(skip)

            if self.downsample and not self.first:
                skip = self.downsample(skip)

        return out + skip


class SelfAttention(nn.Module):
    def __init__(self, in_channel, divider=8):
        super().__init__()

        self.query = spectral_norm(
            nn.Conv2d(in_channel, in_channel // divider, 1, bias=False)
        )
        self.key = spectral_norm(
            nn.Conv2d(in_channel, in_channel // divider, 1, bias=False)
        )
        self.value = spectral_norm(
            nn.Conv2d(in_channel, in_channel // 2, 1, bias=False)
        )
        self.out = spectral_norm(nn.Conv2d(in_channel // 2, in_channel, 1, bias=False))

        self.divider = divider

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        batch, channel, height, width = input.shape

        query = (
            self.query(input)
            .view(batch, channel // self.divider, height * width)
            .transpose(1, 2)
        )
        key = F.max_pool2d(self.key(input), 2).view(
            batch, channel // self.divider, height * width // 4
        )
        value = F.max_pool2d(self.value(input), 2).view(
            batch, channel // 2, height * width // 4
        )
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 2)
        attn = torch.bmm(value, attn.transpose(1, 2)).view(
            batch, channel // 2, height, width
        )
        attn = self.out(attn)
        out = self.gamma * attn + input

        return out


class Generator(nn.Module):
    def __init__(
        self,
        n_class,
        dim_z,
        dim_class,
        feature_channels=(64, 128, 256, 512, 512, 4096, 1000),
        channel_multiplier=64,
        channels=(8, 8, 4, 2, 2, 1),
        blocks='rrrrar',
        upsample='nuuunu',
        activation='relu',
        feature_kernel_size=1,
    ):
        super().__init__()

        self.n_resblock = len([c for c in blocks if c == 'r'])
        self.use_affine = [b == 'r' for b in blocks]

        self.embed = nn.Embedding(n_class, dim_class)

        self.linears = nn.ModuleList()
        self.feature_linears = nn.ModuleList()

        in_dim = dim_z
        feat_i = 6
        for _ in range(2):
            dim = feature_channels[feat_i]

            self.linears.append(
                nn.Sequential(spectral_norm(nn.Linear(in_dim, dim)), nn.LeakyReLU(0.2))
            )

            self.feature_linears.append(spectral_norm(nn.Linear(dim, dim)))

            in_dim = dim
            feat_i -= 1

        self.linear_expand = spectral_norm(
            nn.Linear(in_dim, 7 * 7 * feature_channels[-3])
        )

        self.blocks = nn.ModuleList()
        self.feature_blocks = nn.ModuleList()

        in_channel = channels[0] * channel_multiplier
        for block, ch, up in zip(blocks, channels, upsample):
            if block == 'r':
                self.blocks.append(
                    ResBlock(
                        in_channel,
                        ch * channel_multiplier,
                        norm=True,
                        embed_dim=dim_class,
                        upsample=up == 'u',
                        activation=activation,
                    )
                )

                self.feature_blocks.append(
                    spectral_norm(
                        nn.Conv2d(
                            feature_channels[feat_i],
                            ch * channel_multiplier,
                            feature_kernel_size,
                            padding=(feature_kernel_size - 1) // 2,
                            bias=False,
                        )
                    )
                )

                feat_i -= 1

            elif block == 'a':
                self.blocks.append(SelfAttention(in_channel))

            in_channel = ch * channel_multiplier

        self.colorize = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            get_activation(activation),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channel, 3, 3, padding=1)),
            nn.Tanh(),
        )

    def forward(self, input, class_id, features, masks):
        embed = self.embed(class_id)

        batch_size = input.shape[0]

        feat_i = len(features) - 1

        out = input

        for linear, feat_linear in zip(self.linears, self.feature_linears):
            out = linear(out)

            # print(out.shape, features[feat_i].shape, masks[feat_i].shape)

            out = out + feat_linear(features[feat_i] * masks[feat_i].squeeze(-1))

            # print(out.shape)

            feat_i -= 1

        out = self.linear_expand(out).view(batch_size, -1, 7, 7)

        layer_i = feat_i

        for affine, block in zip(self.use_affine, self.blocks):
            # print(out.shape)
            if affine:
                out = block(out, embed)
                # print(out.shape, features[feat_i].shape, masks[feat_i].shape)
                out = out + self.feature_blocks[layer_i - feat_i](
                    features[feat_i] * masks[feat_i]
                )
                feat_i -= 1

            else:
                out = block(out)

        out = self.colorize(out)

        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        n_class,
        channel_multiplier=64,
        channels=(1, 2, 2, 4, 8, 16, 16),
        blocks='rrarrrr',
        downsample='ddndddn',
        activation='relu',
    ):
        super().__init__()

        blocks_list = []

        in_channel = 3
        for i, (block, ch, down) in enumerate(zip(blocks, channels, downsample)):
            if block == 'r':
                blocks_list.append(
                    ResBlock(
                        in_channel,
                        ch * channel_multiplier,
                        downsample=down == 'd',
                        first=i == 0,
                        activation=activation,
                    )
                )

            elif block == 'a':
                blocks_list.append(SelfAttention(in_channel))

            in_channel = ch * channel_multiplier

        blocks_list += [get_activation(activation)]

        self.blocks = nn.Sequential(*blocks_list)

        self.embed = spectral_norm(nn.Embedding(n_class, in_channel))
        self.linear = spectral_norm(nn.Linear(in_channel, 1))

    def forward(self, input, class_id):
        out = self.blocks(input)

        out = out.sum([2, 3])
        out_linear = self.linear(out)

        embed = self.embed(class_id)
        prod = (out * embed).sum(1, keepdim=True)

        out_linear = out_linear + prod

        return out_linear.squeeze(1)
