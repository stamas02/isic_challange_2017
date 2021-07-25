from src import utils
from src.dataset import ImageData
import pandas as pd
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

DIR_TEST_DATA = "ISIC-2017_Test_v2_Data"
FILE_TEST_LABELS = "ISIC-2017_Test_v2_Part3_GroundTruth.csv"


def test(model_path, dataset_dir, batch_size, image_x, image_y):
    log_name = os.path.basename(model_path).split("-")[0]
    log_dir = os.path.dirname(model_path)
    device = torch.device("cuda")

    model = torch.load(model_path)
    model.eval()
    test_df = pd.read_csv(os.path.join(dataset_dir, FILE_TEST_LABELS))
    test_files = [os.path.join(dataset_dir, DIR_TEST_DATA, f + ".jpg") for f in test_df.image_id]
    test_labels = np.array(test_df.melanoma == 1, dtype=float).reshape((-1, 1))
    test_dataset = ImageData(test_files, test_labels, transform=utils.get_test_transform((image_x, image_y)))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    predictions = []
    files = []
    labels = []
    with torch.no_grad():
        for images, _labels, _files in tqdm(test_data_loader, desc="Predicting on test set"):
            images = images.to(device)
            logits = model(images, dropout=False)
            predictions += torch.sigmoid(logits).detach().cpu().numpy().flatten().tolist()
            files += _files
            labels += _labels.detach().cpu().numpy().flatten().tolist()

    df_test_log = pd.DataFrame(data={"file": files,
                                     "melanoma-p": predictions,
                                     "melanoma-gt": labels})

    df_test_log.to_csv(os.path.join(log_dir, log_name + "-test_result.csv"), index=False, header=True)
    evaluate(test_file=os.path.join(log_dir, log_name + "-test_result.csv"),
             log_dir=log_dir,
             log_name=log_name)


def evaluate(test_file, log_dir, log_name):
    df = pd.read_csv(test_file)
    fpr, tpr, thresholds = roc_curve(y_true=df["melanoma-gt"], y_score=df["melanoma-p"])

    df_roc = pd.DataFrame(data={"Fpr": fpr,
                                "Tpr": tpr,
                                "Thresholds": thresholds})
    df_roc.to_csv(os.path.join(log_dir, log_name + "-roc.csv"), index=False, header=True)
    df_roc.plot(x='Fpr', y='Tpr', title="Melanoma classification ", kind='line')
    plt.savefig(os.path.join(log_dir, log_name + "-roc.pdf"), format="pdf", bbox_inches="tight")
    plt.show()



def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Evaluate model.')
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

    # Testing Arguments
    parser.add_argument("--batch_size", type=int,
                        default=2,
                        help="Integer Value - The sizes of the batches during training.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, time_str)
    os.makedirs(args.log_dir)
    test(**args.__dict__)
