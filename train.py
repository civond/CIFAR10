import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

from model import *
from dataset import *
from utils import *
from resnet50 import *

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 32  
IMAGE_WIDTH = 32

PIN_MEMORY = True
LOAD_MODEL = True

DATA_DIR = "data/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, labels)

        # backward    
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    num_classes = 10
    model = ResNet50(img_channel=3, num_classes=num_classes)

    data_load_time = 0
    training_time = 0
    train_epoch_time = []
    total_time = 0

    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomCrop(size=32, 
                    padding=4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
            )
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomCrop(size=32, 
                    padding=4),
        v2.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
            )
    ])
    
    train = True

    start_time = time.perf_counter()
    loader = get_loader(
        DATA_DIR,
        BATCH_SIZE,
        train_transform,
        NUM_WORKERS,
        train,
        PIN_MEMORY
    )

    end_time = time.perf_counter()
    data_load_time = end_time - start_time

    train=False
    val_loader = get_loader(
        DATA_DIR,
        BATCH_SIZE,
        test_transform,
        NUM_WORKERS,
        train,
        PIN_MEMORY
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL: 
        print("Loading model.")
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch+1}/{NUM_EPOCHS}")

        start_time = time.perf_counter()
        train_fn(loader, 
                 model, 
                 optimizer, 
                 loss_fn, 
                 scaler)
        end_time = time.perf_counter()
        training_time = training_time + (end_time-start_time)
        train_epoch_time.append(training_time)

        training_time = 0 # Reset for next epoch

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)
    total = sum(train_epoch_time)

    print(f"Dataloading time: {data_load_time} ({100*round(data_load_time/(training_time+data_load_time), 2)}%)")
    for i in train_epoch_time:
        print(f"Train time {i+1}: {train_epoch_time[i]} ({100*round(train_epoch_time[i]/(train_epoch_time[i]+data_load_time), 2)}%)")
    print(f"Total time: {total}")

if __name__ == "__main__":
    main()