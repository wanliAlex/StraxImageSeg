import torch
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import torch.nn as nn
from model import UNET
from utils import load_ckpt, save_ckpt, get_loaders, check_accuracy, save_predictions_as_imgs


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALD_SPLIT = 0.25
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LEARNING_RATE = 1e-4
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
TEST_IMG_DIR = "dataset/test_images/"
TEST_MASK_DIR = "dataset/test_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(deivce = DEVICE)
        target = targets.float().unsqueeze(1).to(device = DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Rotate(limit = 35, p = 1.0),
            A. HorizontalFlip(p = 0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels = 3, out_channels = 1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        test_transform,
        NUM_WORKERS
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, scaler)

        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_ckpt(checkpoint)

        check_accuracy(test_loader, model, device=DEVICE)

        #save chpt

if __name__ == "__main__":
    main()
