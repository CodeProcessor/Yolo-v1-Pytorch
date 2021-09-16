import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import YoloDataset
from loss import YoloLoss
from model import YoloV1
from params import SPLIT_SIZE, NO_OF_BOXES, NO_OF_CLASSES, DEVICE, WEIGHT_DECAY, LR, CHANNELS, BATCH_SIZE, NUM_WORKERS, \
    PIN_MEMORY, EPOCHS, PATH, LOAD_MODEL
from utils import (
    mean_average_precision,
    get_bboxes, cellboxes_to_boxes, non_max_suppression, plot_image_custom, save_checkpoint, load_checkpoint
)

seed = 28
torch.manual_seed(seed)

# CHANNELS=1
# NO_OF_CLASSES = 5
# NO_OF_BOXES = 1
# SPLIT_SIZE=7
# PATH='trained_models/best-model-parameters.pt'
#
# LR = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# WEIGHT_DECAY = 0
# EPOCHS = 250
# NUM_WORKERS = 2
# PIN_MEMORY = True
# LOAD_MODEL = False
# LOAD_MODEL_FILE = ''
# IMG_DIR = 'data/screenshots'
# LABEL_DIR = 'data/annotated_data'

torch.autograd.set_detect_anomaly(True)


class Compose(object):
    def __init__(self, _transforms):
        self.transforms = _transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img), boxes

        return img, boxes


transform = Compose([
    transforms.Resize((448, 448)),
    # transforms.rgb
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
loss_func = YoloLoss(split_size=SPLIT_SIZE, num_boxes=NO_OF_BOXES, num_classes=NO_OF_CLASSES)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    mean_average_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {mean_average_loss}")
    return mean_average_loss


def save_images(model, test_loader):
    for i, (x, y) in enumerate(test_loader):
        x = x.to(DEVICE)
        for idx in range(BATCH_SIZE - 1):
            bboxes = cellboxes_to_boxes(model(x), C=NO_OF_CLASSES)
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            # print(bboxes)
            plot_image_custom(x[idx].permute(1, 2, 0).to("cpu"), bboxes,
                              image_name=f"assets/Predicted_Image_{i}_{idx}.jpg")
            # print(idx)


def main():
    model = YoloV1(input_channels=CHANNELS, split_size=SPLIT_SIZE, num_boxes=NO_OF_BOXES, num_classes=NO_OF_CLASSES).to(
        DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    train_dataset = YoloDataset(
        dir_name="train",
        transform=transform,
        S=SPLIT_SIZE, B=NO_OF_BOXES, C=NO_OF_CLASSES
    )

    test_dataset = YoloDataset(
        dir_name="val",
        transform=transform,
        S=SPLIT_SIZE, B=NO_OF_BOXES, C=NO_OF_CLASSES
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(PATH), model, optimizer)

    current_map = 0.9
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}/{EPOCHS}")
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE, C=NO_OF_CLASSES
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NO_OF_CLASSES
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_func
        )

        if mean_avg_prec > current_map:
            print(f"saving best model MAP > {current_map}")
            current_map = mean_avg_prec
            # torch.save(model.state_dict(), PATH)
            checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=PATH)

        save_images(model, test_loader)


if __name__ == '__main__':
    main()
