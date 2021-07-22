import argparse
import os
import zipfile

from src import web_helper

LINK_TRAINING_DATA = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
LINK_TRAINING_LABELS = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv"
LINK_VALIDATION_DATA = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip"
LINK_VALIDATION_LABELS = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv"
LINK_TEST_DATA = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
LINK_TEST_LABELS = "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv"


def main(output):
    output_file = os.path.join(output, "data.zip")

    print("Downloading data. Might take a while.")
    for url in [LINK_TRAINING_DATA, LINK_VALIDATION_DATA, LINK_TEST_DATA]:
        web_helper.download_url(url, output_file)
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            print("Extracting zip file...")
            zip_ref.extractall(output)
        os.remove(output_file)

    print("Downloading ground truth.")
    for url in [LINK_TRAINING_LABELS, LINK_VALIDATION_LABELS, LINK_TEST_LABELS]:
        web_helper.download_url(url, os.path.join(output, os.path.basename(url)))

    pass


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Downloads the ISIC 2017 Challange data.')
    # Dataset Arguments
    parser.add_argument("--output", "-o",
                        type=str,
                        help='String Value - Destination path',
                        )
    args = parser.parse_args()
    return args

 
if __name__ == '__main__':
    args = parseargs()
    main(**args.__dict__)
