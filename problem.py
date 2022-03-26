import os
from random import random
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

np.set_printoptions(threshold=50)
np.random.seed(42)

problem_title = "Food Multi-classification"

_prediction_label_names = [i for i in range(30)]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.SimplifiedImageClassifier(n_classes=len(_prediction_label_names))

# TODO: Check this, the metrics may be chosen more wisely.
# Accuracy, Recall, F1, Precision
# F1 = harmonic mean of precision and recall so we should probably not add recall & precision if we use F1
score_types = [rw.score_types.Accuracy(), rw.score_types.F1Above()]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
    return cv.split(X, y)


_target_column_name = "labels"


def _get_data(path=".", split="train"):
    base_data_path = os.path.abspath(os.path.join(path, "data", split))
    labels_path = os.path.join(base_data_path, "labels.csv")
    labels_df = pd.read_csv(labels_path)
    filepaths = []
    y = np.zeros((len(labels_df.index), len(_prediction_label_names)), dtype=float)
    for j, (i, row) in enumerate(labels_df.iterrows()):
        filename = row["file"]
        # filepath = os.path.join(base_data_path, "images/", filename)
        filepaths.append(filename)

        # 1-hot encoding multiclass multilabel
        file_labels = [int(l) for l in row["labels"].split(';')]
        for i in file_labels:
            y[j, i] = 1.0

    X = np.array(filepaths, dtype=object)

    assert len(X) == len(y)
    # TODO: check that the --quick-test works there
    '''
    if os.environ.get("RAMP_TEST_MODE", False):
        # launched with --quick-test option; only a small subset of the data
        X = X[[1, -1]]
        y = y[[1, -1]]
    '''
    return (os.path.join(base_data_path, "images/"), X), y


def get_train_data(path="."):
    """Get train data from ``data/train/labels.csv``

    Returns
    -------
    X : np.array
        array of shape (N_images,).
        each element in the array is an absolute path to an image
    y : np.array
        array of shape (N_images,).
        each element in the array if a list of variable length.
        each element in this list is a labelled location as a dictionnary::

            {"class": "Primary", "bbox": (2022, 8282, 2300, 9000)}

    """
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")
