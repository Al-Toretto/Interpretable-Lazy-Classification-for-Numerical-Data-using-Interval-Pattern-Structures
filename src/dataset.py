import os


project_root = os.environ.get("PROJECT_ROOT")

if not project_root:
    raise EnvironmentError("PROJECT_ROOT environment variable is not set")


known_datasets = [
    "wine",
    "breast_cancer",
    "rice",
    "sonar",
    "parkinsons",
    "spam",
    "magic",
]

paths_to_datasets = {
    "wine": os.path.join(project_root, "datasets", "wine.csv"),
    "breast_cancer": os.path.join(project_root, "datasets", "breast-cancer.csv"),
    "rice": os.path.join(project_root, "datasets", "rice.csv"),
    "sonar": os.path.join(project_root, "datasets", "sonar.csv"),
    "parkinsons": os.path.join(project_root, "datasets", "parkinsons.data"),
    "spam": os.path.join(project_root, "datasets", "spambase.csv"),
    "magic": os.path.join(project_root, "datasets", "magic.csv"),
}

dataset_positive_labels = {
    "wine": "good",
    "breast_cancer": "M",
    "rice": "Osmancik",
    "sonar": "M",
    "parkinsons": 1,
    "spam": 1,
    "magic": "g",
}
dataset_class_column_names = {
    "wine": "quality",
    "breast_cancer": "diagnosis",
    "rice": "class",
    "sonar": "class",
    "parkinsons": "status",
    "spam": "class",
    "magic": "class",
}


class Dataset:
    def __init__(
        self,
        dataset_name=None,
        dataset_path=None,
        class_column_name=None,
        positive_label=None,
    ):
        self.dataset_name = None
        self.dataset_path = None
        self.class_column_name = None
        self.dataset_positive_label = None
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        if dataset_name is not None and dataset_name in known_datasets:
            self.dataset_name = dataset_name
            self.dataset_path = paths_to_datasets[dataset_name]
            self.dataset_class_column_name = dataset_class_column_names[
                dataset_name
            ]
            self.dataset_positive_label = dataset_positive_labels[dataset_name]
        
        else:
            self.dataset_path = dataset_path
            self.dataset_class_column_name = class_column_name
            self.dataset_positive_label = positive_label
