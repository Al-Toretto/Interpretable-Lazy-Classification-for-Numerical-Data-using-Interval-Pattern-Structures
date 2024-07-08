from src.dataset import Dataset#, known_datasets
from src.dataset_preprocessor import DatasetPreprocessor
from src.ips_knn_classifier import IPSKNNClassifier
from src.hyperparameter_tuner import HyperparameterTuner
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from src.utils import custom_print


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


known_datasets = [
    "glass",
    "ionosphere",
    "page_blocks",
    "waveform",
]


param_grid_dict = {
    "ips_knn": {"k": list(range(1, 101))},
    "knn": {"n_neighbors": list(range(1, 101)), "weights": ["uniform", "distance"]},
    "naive_bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    "logistic_regression": {
        "penalty": ["l1", "l2"],
        "C": [10**x for x in range(-5, 5, 1)],
        "solver": ["liblinear", 'saga' , 'lbfgs'],
    },
    "svm": {
        "C": [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"] + [0.001, 0.01, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
    },
    "decision_tree": {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2],
  "max_features": ["sqrt", "log2"],
  "min_impurity_decrease": [0.0, 0.1]
    },
    "random_forest": {
        "n_estimators": [10, 20, 50, 100, 200, 500, 1000],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", None],
        "min_impurity_decrease": [0.0, 0.1],
  "bootstrap": [True, False],
    },
    "xgboost": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.1, 0.01, 0.05],
        "n_estimators": [10, 20, 50, 100, 200, 500, 1000],
        "subsample": [0.5, 0.8],
        "colsample_bytree": [0.5, 0.8],
        "scale_pos_weight": [1, 3],
        "reg_alpha": [0, 1e-5],
        "reg_lambda": [0, 1e-5],
    },
}





def main():
    filename="exp_hyperparameter_search.txt"
    for name in known_datasets:
        custom_print(f"Dataset ------------ {name} ------------------------------------", filename)
        custom_print("------------------------------------------------------------------------------", filename)
        dataset = Dataset(dataset_name=name)
        preprocessor = DatasetPreprocessor(dataset).preprocess()
        if len(dataset.y.unique()) > 2:
            scorer = make_scorer(f1_score, average='macro')
        else:
            scorer = make_scorer(f1_score, pos_label=dataset.dataset_positive_label)

        # # Naive Bayes: ******************************************************************
        custom_print('Naive Bayes: ******************************************************************', filename)
        classifier = GaussianNB()
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['naive_bayes'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)


        # XGBoost: **********************************************************************
        custom_print('XGBoost: **********************************************************************', filename)
        pos_label = dataset.dataset_positive_label
        
        if len(dataset.y.unique()) == 2:
            transfered_y_train = dataset.y_train.apply(lambda x: 1 if x == pos_label else 0)
            transfered_y_test = dataset.y_test.apply(lambda x: 1 if x == pos_label else 0)
            xg_scorer = make_scorer(f1_score, pos_label=1)
            pos_label = 1
        else:
            label_encoder = LabelEncoder()
            transfered_y_train = label_encoder.fit_transform(dataset.y_train)
            transfered_y_test = label_encoder.fit_transform(dataset.y_test)
            xg_scorer = make_scorer(f1_score, average='macro')
        
        classifier = xgb.XGBClassifier(random_state=1998)
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['xgboost'])

        tuner.perform_stratified_grid_search(dataset.X_train, transfered_y_train, scoring=xg_scorer, verbose=1, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(transfered_y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(transfered_y_test, y_pred, pos_label=pos_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(transfered_y_test, y_pred)}", filename)
        custom_print("\n\n", filename)
        


        preprocessor = preprocessor.standardize()
        # IPS-KNN: **********************************************************************
        custom_print('IPS-KNN: **********************************************************************', filename)
        classifier = IPSKNNClassifier()
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['ips_knn'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)
      
        # KNN: **************************************************************************
        custom_print('KNN: **************************************************************************', filename)
        classifier = KNeighborsClassifier()
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['knn'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)

        # Logistic Regression: **********************************************************
        custom_print('Logistic Regression: **********************************************************', filename)
        classifier = LogisticRegression(random_state=1998)
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['logistic_regression'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)

        # SVM: **************************************************************************
        custom_print('SVM: **************************************************************************', filename)
        classifier = SVC(random_state=1998)
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['svm'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)

        # Decision Tree: ****************************************************************
        custom_print('Decision Tree: ****************************************************************', filename)
        classifier = DecisionTreeClassifier(random_state=1998)
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['decision_tree'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=0, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)

        # Random Forest: ****************************************************************
        custom_print('Random Forest: ****************************************************************', filename)
        classifier = RandomForestClassifier(random_state=1998)
        tuner = HyperparameterTuner(estimator=classifier, param_grid=param_grid_dict['random_forest'])
        tuner.perform_stratified_grid_search(dataset.X_train, dataset.y_train, scoring=scorer, verbose=1, n_jobs=-1)
        custom_print(f"Tuner's best parameters: {tuner.best_params_}", filename)
        custom_print(f"Tuner's best F1 score: {tuner.best_score_}", filename)
        best_classifier = tuner.best_estimator_
        y_pred = best_classifier.predict(dataset.X_test)
        custom_print(
            f"X_test F1 Score: {f1_score(dataset.y_test, y_pred, average='macro') if len(dataset.y.unique()) > 2 else f1_score(dataset.y_test, y_pred, pos_label=dataset.dataset_positive_label)}"
        , filename)
        custom_print(f"X_test accuracy: {accuracy_score(dataset.y_test, y_pred)}", filename)
        custom_print("\n\n", filename)

if __name__ == "__main__":
    main()
