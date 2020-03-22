import os

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.png',
    '.ppm',
    '.bmp',
    '.pgm',
    '.tif',
    '.tiff',
    '.webp',
)


class Places365(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.data = []
        categories = set()

        for dirpath, dirnames, filenames in os.walk(self.root):
            if len(filenames) > 0:
                relpath = os.path.relpath(dirpath, root)
                _, category = os.path.split(dirpath)
                categories.add(category)

                for file in filenames:
                    if file.lower().endswith(IMG_EXTENSIONS):
                        self.data.append((os.path.join(relpath, file), category))

        categories = sorted(list(categories))

        self.category_map = {cat: i for i, cat in enumerate(categories)}

        self.n_class = 365

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, category = self.data[index]

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        label = self.category_map[category]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
