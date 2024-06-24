from src.dataset import Dataset, known_datasets
from src.dataset_preprocessor import DatasetPreprocessor
from src.ips_knn_classifier import IPSKNNClassifier
from src.eager_ips_knn_classifier import EagerIPSKNNClassifier
from config.optimal_hyperparameters import optimal_params
from src.utils import custom_print

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import time
import numpy as np
from typing import Any
import pandas as pd


def analyze_performance(
    classifier: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_positive_label: Any,
    filename: str,
):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    custom_print("\n\n", filename)
    custom_print(
        f"X_test F1 Score: {f1_score(y_test, y_pred, pos_label=dataset_positive_label)}",
        filename,
    )
    custom_print(f"X_test accuracy: {accuracy_score(y_test, y_pred)}", filename)
    custom_print("\n\n", filename)


def main():
    filename = "exp_f1.txt"
    for name in known_datasets:
        custom_print(
            f"Dataset ------------ {name} ------------------------------------",
            filename,
        )
        custom_print(
            "------------------------------------------------------------------------------",
            filename,
        )
        dataset = Dataset(dataset_name=name)
        preprocessor = DatasetPreprocessor(dataset).preprocess()

        # # Naive Bayes: ******************************************************************
        custom_print(
            "Naive Bayes: ******************************************************************",
            filename,
        )
        classifier = GaussianNB(**optimal_params["naive_bayes"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # XGBoost: **********************************************************************
        custom_print(
            "XGBoost: **********************************************************************",
            filename,
        )
        pos_label = dataset.dataset_positive_label
        transfered_y_train = dataset.y_train.apply(lambda x: 1 if x == pos_label else 0)
        transfered_y_test = dataset.y_test.apply(lambda x: 1 if x == pos_label else 0)
        classifier = xgb.XGBClassifier(
            random_state=1998, n_jobs=1, **optimal_params["xgboost"][name]
        )
        analyze_performance(
            classifier,
            dataset.X_train,
            transfered_y_train,
            dataset.X_test,
            transfered_y_test,
            1,
            filename,
        )

        preprocessor = preprocessor.standardize()

        # # IPS-KNN: **********************************************************************
        custom_print(
            "IPS-KNN: **********************************************************************",
            filename,
        )
        classifier = IPSKNNClassifier(**optimal_params["ips_knn"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # E-IPS-KNN T: ******************************************************************
        custom_print(
            "E-IPS-KNN T: ******************************************************************",
            filename,
        )
        classifier = EagerIPSKNNClassifier(**optimal_params["e_ips_knn_t"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # E-IPS-KNN T&F: ****************************************************************
        custom_print(
            "E-IPS-KNN T&F: ****************************************************************",
            filename,
        )
        classifier = EagerIPSKNNClassifier(**optimal_params["e_ips_knn_tf"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )
        custom_print("\n\n", filename)

        # # E-IPS-KNN T with expansion: ***************************************************
        custom_print(
            "E-IPS-KNN T with expansion: ***************************************************",
            filename,
        )
        classifier = EagerIPSKNNClassifier(**optimal_params["e_ips_knn_t_e"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # E-IPS-KNN T&F with expanstion: ************************************************
        custom_print(
            "E-IPS-KNN T&F with expanstion: ************************************************",
            filename,
        )
        classifier = EagerIPSKNNClassifier(**optimal_params["e_ips_knn_tf_e"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # KNN: **************************************************************************
        custom_print(
            "KNN: **************************************************************************",
            filename,
        )
        classifier = KNeighborsClassifier(n_jobs=1, **optimal_params["knn"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # Logistic Regression: **********************************************************
        custom_print(
            "Logistic Regression: **********************************************************",
            filename,
        )
        classifier = LogisticRegression(
            random_state=1998, n_jobs=1, **optimal_params["logistic_regression"][name]
        )
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # SVM: **************************************************************************
        custom_print(
            "SVM: **************************************************************************",
            filename,
        )
        classifier = SVC(random_state=1998, **optimal_params["svm"][name])
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # Decision Tree: ****************************************************************
        custom_print(
            "Decision Tree: ****************************************************************",
            filename,
        )
        classifier = DecisionTreeClassifier(
            random_state=1998, **optimal_params["decision_tree"][name]
        )
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )

        # # Random Forest: ****************************************************************
        custom_print(
            "Random Forest: ****************************************************************",
            filename,
        )
        classifier = RandomForestClassifier(
            random_state=1998, n_jobs=1, **optimal_params["random_forest"][name]
        )
        analyze_performance(
            classifier,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            dataset.dataset_positive_label,
            filename,
        )


if __name__ == "__main__":
    main()
