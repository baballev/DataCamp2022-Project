import pandas as pd
import os
import urllib.request
import socket
import tqdm
import sys
import shutil

socket.setdefaulttimeout(20)

URI_CSV_PATH = "./data_location_public.csv"
#       split        -          uri            -                labels
# "train" or "test"  -  uri_resource_name.jpg  -  coma separated labels: 1, 3, 18, 22
DATA_PATH = "./data/"
IMAGES_PATH = "images/"
LABEL_PATH = "labels.csv"
TRAIN_PATH = "train/"
TEST_PATH = "test/"

ROOT_URI = "https://images.openfoodfacts.org/images/products/"


def idx_to_str(idx):
    tmp_idx = idx
    zeros = 6
    while tmp_idx // 10 > 0:
        tmp_idx = tmp_idx // 10
        zeros -= 1
    return "0" * zeros + str(idx)


def clean_mess(exit=True):
    print("Files will be cleaned.")
    if os.path.exists(os.path.join(DATA_PATH, TRAIN_PATH)):
        shutil.rmtree(os.path.join(DATA_PATH, TRAIN_PATH))
    if os.path.exists(os.path.join(DATA_PATH, TEST_PATH)):
        shutil.rmtree(os.path.join(DATA_PATH, TEST_PATH))
    if exit: sys.exit(1)


def check_data_integrity():
    # TODO
    pass


clean_mess(exit=False)
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(os.path.join(DATA_PATH, TRAIN_PATH)):
    os.mkdir(os.path.join(DATA_PATH, TRAIN_PATH))
if not os.path.exists(os.path.join(DATA_PATH, TRAIN_PATH, IMAGES_PATH)):
    os.mkdir(os.path.join(DATA_PATH, TRAIN_PATH, IMAGES_PATH))
if not os.path.exists(os.path.join(DATA_PATH, TEST_PATH)):
    os.mkdir(os.path.join(DATA_PATH, TEST_PATH))
if not os.path.exists(os.path.join(DATA_PATH, TEST_PATH, IMAGES_PATH)):
    os.mkdir(os.path.join(DATA_PATH, TEST_PATH, IMAGES_PATH))

df = pd.read_csv(URI_CSV_PATH)
train_idx, test_idx = 0, 0
label_train_df, label_test_df = pd.DataFrame(columns=["file", "labels"]), pd.DataFrame(columns=["file", "labels"])

for idx, row in tqdm.tqdm(df.iterrows(), total=55000):
    uri = ROOT_URI + row["uri"]
    try:
        if row["split"] == "train":
            name = idx_to_str(train_idx) + ".jpg"
            urllib.request.urlretrieve(uri, os.path.join(DATA_PATH, TRAIN_PATH, IMAGES_PATH, name))
            label_train_df.loc[train_idx] = {"file": name, "labels": row["labels"]}

            train_idx += 1
        elif row["split"] == "test":
            name = idx_to_str(test_idx) + ".jpg"
            urllib.request.urlretrieve(uri, os.path.join(DATA_PATH, TEST_PATH, IMAGES_PATH, name))
            label_test_df.loc[train_idx] = {"file": name, "labels": row["labels"]}

            test_idx += 1
        else:
            raise NotImplementedError
    except Exception as e:
        print("An error occured while downloading data:")
        print(e)
        continue

try:
    label_train_df.to_csv(os.path.join(DATA_PATH, TRAIN_PATH, LABEL_PATH))
    label_test_df.to_csv(os.path.join(DATA_PATH, TEST_PATH, LABEL_PATH))
except Exception as e:
    print("An error occured while trying to write csv label files.")
    print(e)
    clean_mess()

print("All data downloaded successfully.")
