import argparse
import os
import sys

# Determine the project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['PROJECT_ROOT'] = project_root

# Append the project root to sys.path
sys.path.append(project_root)


from sklearn.metrics import accuracy_score, f1_score, make_scorer  # noqa: E402


from src.ips_knn_classifier import IPSKNNClassifier # noqa: E402
from src.eager_ips_knn_classifier import EagerIPSKNNClassifier # noqa: E402
from src.dataset import Dataset # noqa: E402
from src.dataset_preprocessor import DatasetPreprocessor # noqa: E402
from src.hyperparameter_tuner import HyperparameterTuner # noqa: E402
from src.utils import custom_print # noqa: E402



def main(path_to_csv, label_column_name, positive_label, model_type):
    filename='classification_results.txt'
    dataset = Dataset("user_provided", path_to_csv, label_column_name, positive_label)
    preprocessor = DatasetPreprocessor(dataset).preprocess().standardize()  # noqa: F841
    scorer = make_scorer(f1_score, pos_label=dataset.dataset_positive_label)

    if model_type == 'lazy':
        classifier = IPSKNNClassifier()
        param_grid = {"k": [i for i in range(1, 101)]}
    elif model_type == 'eager':
        classifier = EagerIPSKNNClassifier(dataset_positive_label=dataset.dataset_positive_label, use_counter_explanations=False, use_hyperrectangle_expanstion_by_information_gain=False)
        param_grid = {
                "k": [i for i in range(1, 31)],
                "dataset_positive_label": [dataset.dataset_positive_label],
                "use_hyperrectangle_expanstion_by_information_gain": [False],
                "use_counter_explanations" : [False, True],
            }
    else:
        raise ValueError("Invalid model type. Choose 'lazy' or 'eager'.")
    
    tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid)
    tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
    custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
    custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
    best_classifier = tuner.best_estimator_
    y_pred = best_classifier.predict(dataset.X_test)
    custom_print(f"{y_pred=}", filename)
    custom_print(
        f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
    , filename)
    custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
    custom_print("\n\n", filename)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify user-provided datasets.')
    parser.add_argument('path_to_csv', type=str, help='Path to the CSV file containing the dataset.')
    parser.add_argument('label_column_name', type=str, help='Name of the column containing the labels.')
    parser.add_argument('positive_label', type=str, help='Value of the positive label.')
    parser.add_argument('model_type', type=str, choices=['lazy', 'eager'], help='Model type to use for classification: "lazy" or "eager".')

    args = parser.parse_args()

    main(args.path_to_csv, args.label_column_name, args.positive_label, args.model_type)
