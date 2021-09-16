

import os

import torch
from PIL import Image

from params import IMG_DIR, LABEL_DIR


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name, S=7, B=1, C=20, transform=None):
        self.img_dir = os.path.join(IMG_DIR, dir_name)
        self.label_dir = os.path.join(LABEL_DIR, dir_name)
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _image_name = self.images[index]
        _name = _image_name.split('.')[0]
        label_path = os.path.join(self.label_dir, f"{_name}.txt")
        boxes = []
        with open(label_path, 'r') as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace('\n', '').split()
                ]
                boxes.append([class_label, x, y, width, height])

        image_path = os.path.join(self.img_dir, _image_name)
        image = Image.open(image_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


if __name__ == '__main__':
    ds = YoloDataset()
