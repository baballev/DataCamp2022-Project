
problem_title = "Products Multi-classification"


##TODO: Change function, just copied them from follicle detection challenge
def get_cv(X, y):
    """Split data by ovary number

    Uses LeaveOneGroupOut where each group is a set of images
    that correspond to a given overy number.

    Parameters:
    -----------
    X : np.array
        array of image absolute paths
    y : np.array
        array of lists of true follicule locations

    """

    def extract_ovary_number(filename):
        digit = re.match(r".*M0(\d)-\d.*", filename).group(1)
        return int(digit)

    groups = [extract_ovary_number(filename) for filename in X]
    cv = LeaveOneGroupOut()
    return cv.split(X, y, groups)


def _get_data(path=".", split="train"):
    """
    Returns
    X : np.array
        shape (N_images,)
    y : np.array
        shape (N_images,). Each element is a list of locations.

    """
    base_data_path = os.path.abspath(os.path.join(path, "data", split))
    labels_path = os.path.join(base_data_path, "labels.csv")
    labels = pd.read_csv(labels_path)
    filepaths = []
    locations = []
    for filename, group in labels.groupby("filename"):
        filepath = os.path.join(base_data_path, filename)
        filepaths.append(filepath)

        locations_in_image = [
            {
                "bbox": (row["xmin"], row["ymin"], row["xmax"], row["ymax"]),
                "class": row["class"],
            }
            for _, row in group.iterrows()
        ]
        locations.append(locations_in_image)

    X = np.array(filepaths, dtype=object)
    y = np.array(locations, dtype=object)
    assert len(X) == len(y)
    if os.environ.get("RAMP_TEST_MODE", False):
        # launched with --quick-test option; only a small subset of the data
        X = X[[1, -1]]
        y = y[[1, -1]]
    return X, y
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
