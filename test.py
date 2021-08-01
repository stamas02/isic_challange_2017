from src import utils
from src.dataset import ImageData
import pandas as pd
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import sklearn.metrics as metrics
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

    _nevus = test_df.melanoma + test_df.seborrheic_keratosis == 0
    test_labels = np.stack([test_df.melanoma, test_df.seborrheic_keratosis, _nevus], axis=1)

    test_dataset = ImageData(test_files, test_labels, transform=utils.get_test_transform((image_x, image_y)))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    predictions = []
    files = []
    labels = []
    with torch.no_grad():
        for images, _labels, _files in tqdm(test_data_loader, desc="Predicting on test set"):
            images = images.to(device)
            logits = model(images, dropout=False)
            predictions += torch.softmax(logits, 1).detach().cpu().numpy().tolist()
            files += _files
            labels += _labels.detach().cpu().numpy().tolist()

    predictions = np.array(predictions)
    labels = np.array(labels)
    df_test_log = pd.DataFrame(data={"file": files,
                                     "melanoma": predictions[:, 0],
                                     "seborrheic_keratosis": predictions[:, 1],
                                     "nevus": predictions[:, 2],
                                     "melanoma_true": labels[:, 0],
                                     "seborrheic_keratosis_true": labels[:, 1],
                                     "nevus_true": labels[:, 2], })

    df_test_log.to_csv(os.path.join(log_dir, log_name + "-test_result.csv"), index=False, header=True)
    evaluate(test_file=os.path.join(log_dir, log_name + "-test_result.csv"),
             log_dir=log_dir,
             log_name=log_name)


def evaluate(test_file, log_dir, log_name):
    df = pd.read_csv(test_file)
    melanoma_p = np.array(df["melanoma"])
    seborrheic_keratosis_p = np.array(df["seborrheic_keratosis"])
    nevus_p = np.array(df["nevus"])

    melanoma_gt = np.array(df["melanoma_true"], dtype=bool)
    seborrheic_keratosis_gt = np.array(df["seborrheic_keratosis_true"], dtype=bool)
    nevus_gt = np.array(df["nevus_true"], dtype=bool)

    results = [(melanoma_gt, melanoma_p, "MEL"),
               (seborrheic_keratosis_gt, seborrheic_keratosis_p, "SK"),
               (nevus_gt, nevus_p, "NV")]

    # ROC
    roc_file_path = os.path.join(log_dir, log_name)
    fig = plt.figure()
    axes = None
    for gt, p, name in results:
        fpr, tpr, thresholds = metrics.roc_curve(y_true=gt, y_score=p)
        df_roc = pd.DataFrame(data={"Fpr": fpr, "Tpr": tpr, "Thresholds": thresholds})
        df_roc.to_csv(f"{roc_file_path}-{name}-roc.csv", index=False, header=True)
        plt.plot(fpr, tpr, label=name)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(f"{roc_file_path}roc.pdf", format="pdf", bbox_inches="tight")

    # Integral Metrics
    auc = [metrics.roc_auc_score(gt, p) for gt, p, _ in results]
    auc_80 = [metrics.roc_auc_score(gt, p, max_fpr=(1 - 0.8)) for gt, p, _ in results]
    avg_precision = [metrics.average_precision_score(gt, p) for gt, p, _ in results]

    # Threshold Metrics
    threshold = 0.5
    cn_matrices = np.array([metrics.confusion_matrix(gt, p >= 0.5).ravel() for gt, p, _ in results])
    tn, fp, fn, tp = cn_matrices[:, 0], cn_matrices[:, 1], cn_matrices[:, 2], cn_matrices[:, 3]

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (tpr + tnr) / 2
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    dice = [metrics.f1_score(gt, p >= 0.5) for gt, p, _ in results]
    jaccard = [metrics.jaccard_score(gt, p >= 0.5) for gt, p, _ in results]

    df_performance = pd.DataFrame(data={"Metrics": [name for _, _, name in results],
                                        "AUC": auc,
                                        "AUC, Sens > 80%": auc_80,
                                        "Average Precision": avg_precision,
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": balanced_accuracy,
                                        "Sensitivity": tpr,
                                        "Specificity": tnr,
                                        "Dice Coefficient": dice,
                                        "Jaccard Index": jaccard,
                                        "PPV": ppv,
                                        "NPV": npv})

    df_performance.to_csv(f"{roc_file_path}-performance.csv", index=False, header=True)


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
