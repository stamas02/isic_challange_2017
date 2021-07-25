from src import utils
from src.dataset import ImageData
from src.model import EfficientNetClassifier
from test import test
import pandas as pd
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import argparse
import os
import numpy as np
from tqdm import tqdm
import time

DIR_TRAINING_DATA = "ISIC-2017_Training_Data"
FILE_TRAINING_LABELS = "ISIC-2017_Training_Part3_GroundTruth.csv"
DIR_VALIDATION_DATA = "ISIC-2017_Validation_Data"
FILE_VALIDATION_LABELS = "ISIC-2017_Validation_Part3_GroundTruth.csv"


def read_datasets(dataset_files):
    df = pd.DataFrame()
    for dataset_file in dataset_files:
        _df = pd.read_csv(dataset_file)
        df = pd.concat([df, _df], ignore_index=True)
    return df


def train(dataset_dir, image_x, image_y, lr, lr_decay, lr_step, batch_size, epoch, log_dir, log_name, do_test):
    train_df = pd.read_csv(os.path.join(dataset_dir, FILE_TRAINING_LABELS))
    val_df = pd.read_csv(os.path.join(dataset_dir, FILE_VALIDATION_LABELS))

    train_files = [os.path.join(dataset_dir, DIR_TRAINING_DATA, f + ".jpg") for f in train_df.image_id]
    val_files = [os.path.join(dataset_dir, DIR_VALIDATION_DATA, f + ".jpg") for f in val_df.image_id]

    train_labels = np.array(train_df.melanoma == 1, dtype=float).reshape((-1, 1))
    val_labels = np.array(val_df.melanoma == 1, dtype=float).reshape((-1, 1))

    train_dataset = ImageData(train_files, train_labels, transform=utils.get_train_transform((image_x, image_y)))
    val_dataset = ImageData(val_files, val_labels, transform=utils.get_test_transform((image_x, image_y)))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    device = torch.device("cuda")
    model = EfficientNetClassifier(b=0, num_classes=1).to(device)
    optimizer = SGD(params=model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

    df_train_log = pd.DataFrame(columns=['epoch', 'train-loss', 'val-loss'])

    for _epoch in range(epoch):
        model.train()
        train_loss = 0
        p_bar = tqdm(train_data_loader, desc=f"Training epoch {_epoch}")
        for i, (images, labels, _) in enumerate(p_bar):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, dropout=True)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            train_loss = train_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))
            p_bar.set_postfix({'loss': train_loss})
            optimizer.step()

        scheduler.step()
        model.eval()
        val_loss = 0
        p_bar = tqdm(val_data_loader, desc=f"Validation epoch {_epoch}")
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(p_bar):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images, dropout=False)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                val_loss = val_loss * (1 - (1 / (i + 1))) + loss.item() * (1 / (i + 1))

        df_train_log = df_train_log.append({'epoch': _epoch,
                                            'train-loss': train_loss,
                                            'val-loss': val_loss}, ignore_index=True)

    df_train_log.to_csv(os.path.join(log_dir, log_name + "-train_log.csv"), index=False, header=True)
    torch.save(model, os.path.join(log_dir, log_name + "-model.pt"))

    if do_test:
        test(model_path=os.path.join(log_dir, log_name + "-model.pt"),
             dataset_dir=dataset_dir,
             batch_size=batch_size,
             image_x=image_x,
             image_y=image_y)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate different thresholds')
    # Dataset Arguments
    parser.add_argument("--dataset-dir", "-d",
                        type=str,
                        help='String Value - The folder where the dataset is downloaded using get_dataset.py',
                        )
    parser.add_argument("--image_x", type=int,
                        default=300,
                        help="Integer Value - Width of the image that should be resized to.")
    parser.add_argument("--image_y", type=int,
                        default=225,
                        help="Integer Value - Height of the image that should be resized to.")

    # Training Arguments
    parser.add_argument("--lr", type=float,
                        default=0.1,
                        help="Floating Point Value - Starting learning rate.")
    parser.add_argument("--lr_decay", type=float,
                        default=0.5,
                        help="Floating Point Value - Learning rate decay.")
    parser.add_argument("--lr_step", type=float,
                        default=15,
                        help="Integer Value - Decay learning rate after stepsize.")
    parser.add_argument("--batch_size", type=int,
                        default=2,
                        help="Integer Value - The sizes of the batches during training.")
    parser.add_argument("--epoch", type=int,
                        default=0,
                        help="Integer Value - Number of epoch.")
    parser.add_argument('--do-test', action='store_true',
                        help="Flag Boolean Value - If set testing will be carried out after training")

    # Logging Arguments
    parser.add_argument("--log-dir", type=str,
                        help="String Value - Path to the folder the log is to be saved.")
    parser.add_argument("--log-name", type=str,
                        default=225,
                        help="String Value - This is a descriptive name of the method. "
                             "Will be used in legends e.g. ROC curve")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, time_str)
    os.makedirs(args.log_dir)
    train(**args.__dict__)
