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
import time
import numpy as np



def measure_avg_wall_time(classifier, X_train, y_train, X_test, filename, n_repetitions = 10):
    fit_times = []
    predict_times = []
    for _ in range(n_repetitions):
        start = time.perf_counter()
        classifier.fit(X_train, y_train)
        end = time.perf_counter()
        fit_times.append(end - start)

        start = time.perf_counter()
        classifier.predict(X_test)
        end = time.perf_counter()
        predict_times.append(end - start)

    custom_print(f"AVG fit time: {np.average(fit_times)}", filename)
    custom_print(f"AVG predict time: {np.average(predict_times)}", filename)


def measure_avg_wall_time_ips_knn_based(classifier, X_train, y_train, X_test, filename: str, n_repetitions = 10):
    fit_times = []
    predict_times = []
    rc_times = []
    rrc_times = []
    fi_times = []
    cf_times = []

    for i in range(n_repetitions):
        print(i)
        start = time.perf_counter()
        classifier.fit(X_train, y_train)
        end = time.perf_counter()
        fit_times.append(end - start)

        start = time.perf_counter()
        classifier.predict(X_test)
        end = time.perf_counter()
        predict_times.append(end - start)

        start = time.perf_counter()
        classifier._predict_with_explanation(X_test, False, False, False, False)
        end = time.perf_counter()
        rc_times.append(end - start - predict_times[-1])

        start = time.perf_counter()
        classifier._predict_with_explanation(X_test, True, False, False, False)
        end = time.perf_counter()
        rrc_times.append(end - start - predict_times[-1] - rc_times[-1])

        start = time.perf_counter()
        classifier._predict_with_explanation(X_test, False, True, False, False)
        end = time.perf_counter()
        fi_times.append(end - start - predict_times[-1] - rc_times[-1])

        start = time.perf_counter()
        classifier._predict_with_explanation(X_test,  False, False, True, False)
        end = time.perf_counter()
        cf_times.append(end - start - predict_times[-1] - rc_times[-1])

    custom_print(f"AVG fit time: {np.average(fit_times)}", filename)
    custom_print(f"AVG predict time: {np.average(predict_times)}", filename)
    custom_print(f"AVG time for finding reason for classification: {np.average(rc_times)}", filename)
    custom_print(f"AVG time for finding reduced reason for classification: {np.average(rrc_times)}", filename)
    custom_print(f"AVG time for finding feature imporatnce: {np.average(fi_times)}", filename)
    custom_print(f"AVG time for finding counterfactual explanations: {np.average(cf_times)}", filename)
    custom_print(f"AVG explanation time: {np.average(rc_times) + np.average(rrc_times) + np.average(fi_times) + np.average(cf_times)}", filename)


def main():
    filename = 'exp_time_effecieny.txt'
    for name in known_datasets:
        custom_print(f"Dataset ------------ {name} ------------------------------------", filename)
        custom_print("------------------------------------------------------------------------------", filename)
        dataset = Dataset(dataset_name=name)
        preprocessor = DatasetPreprocessor(dataset).preprocess()


        # # Naive Bayes: ******************************************************************
        custom_print('Naive Bayes: ******************************************************************', filename)
        classifier = GaussianNB(**optimal_params['naive_bayes'][name])
        measure_avg_wall_time(classifier, dataset.X_train, dataset.y_train, dataset.X_test, filename)
        custom_print("\n\n", filename)


        # # XGBoost: **********************************************************************
        custom_print('XGBoost: **********************************************************************', filename)
        pos_label = dataset.dataset_positive_label
        transfered_y_train = dataset.y_train.apply(lambda x: 1 if x == pos_label else 0)
        classifier = xgb.XGBClassifier(random_state=1998, n_jobs=1, **optimal_params['xgboost'][name])
        
        measure_avg_wall_time(classifier, dataset.X_train, transfered_y_train, dataset.X_test, filename)
        custom_print("\n\n", filename)

        preprocessor = preprocessor.standardize()

        # # IPS-KNN: **********************************************************************
        custom_print('IPS-KNN: **********************************************************************', filename)
        classifier = IPSKNNClassifier(**optimal_params['ips_knn'][name])
        classifier.fit(dataset.X_train, dataset.X_test)
        classifier.destandardize_features(preprocessor)
        measure_avg_wall_time_ips_knn_based(classifier, dataset.X_train, dataset.y_train, dataset.X_test, filename)
        custom_print("\n\n", filename)

        # E-IPS-KNN T: ******************************************************************
        custom_print('E-IPS-KNN T: ******************************************************************', filename)
        classifier = EagerIPSKNNClassifier(**optimal_params['e_ips_knn_t'][name])
        # classifier.destandardize_hyperrectangles(preprocessor)
        measure_avg_wall_time_ips_knn_based(classifier, dataset.X_train, dataset.y_train, dataset.X_test, filename)
        custom_print("\n\n", filename)

        # # E-IPS-KNN T&F: ****************************************************************
        custom_print('E-IPS-KNN T&F: ****************************************************************', filename)
        classifier = EagerIPSKNNClassifier(**optimal_params['e_ips_knn_tf'][name])
        # classifier.destandardize_hyperrectangles(preprocessor)
        measure_avg_wall_time_ips_knn_based(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        custom_print("\n\n", filename)

        # # E-IPS-KNN T with expansion: ***************************************************
        # custom_print('E-IPS-KNN T with expansion: ***************************************************', filename)
        # classifier = EagerIPSKNNClassifier(**optimal_params['e_ips_knn_t_e'][name])
        # # classifier.destandardize_hyperrectangles(preprocessor)
        # measure_avg_wall_time_ips_knn_based(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        # custom_print("\n\n", filename)

        # E-IPS-KNN T&F with expanstion: ************************************************
        # custom_print('E-IPS-KNN T&F with expanstion: ************************************************', filename)
        # classifier = EagerIPSKNNClassifier(**optimal_params['e_ips_knn_tf_e'][name])
        # # classifier.destandardize_hyperrectangles(preprocessor)
        # measure_avg_wall_time_ips_knn_based(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        # custom_print("\n\n", filename)

        # # KNN: **************************************************************************
        custom_print('KNN: **************************************************************************', filename)
        classifier = KNeighborsClassifier(n_jobs=1, **optimal_params['knn'][name])
        measure_avg_wall_time(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        custom_print("\n\n", filename)

        # # Logistic Regression: **********************************************************
        custom_print('Logistic Regression: **********************************************************', filename)
        classifier = LogisticRegression(random_state=1998, n_jobs=1, **optimal_params['logistic_regression'][name])
        measure_avg_wall_time(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        custom_print("\n\n", filename)

        # # SVM: **************************************************************************
        custom_print('SVM: **************************************************************************', filename)
        classifier = SVC(random_state=1998, **optimal_params['svm'][name])
        measure_avg_wall_time(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        custom_print("\n\n", filename)

        # # Decision Tree: ****************************************************************
        custom_print('Decision Tree: ****************************************************************', filename)
        classifier = DecisionTreeClassifier(random_state=1998, **optimal_params['decision_tree'][name])
        measure_avg_wall_time(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        custom_print("\n\n", filename)

        # # Random Forest: ****************************************************************
        custom_print('Random Forest: ****************************************************************', filename)
        classifier = RandomForestClassifier(random_state=1998, n_jobs=1, **optimal_params['random_forest'][name])
        measure_avg_wall_time(classifier, dataset.X_train, dataset.y_train, dataset.X_test)
        custom_print("\n\n", filename)


if __name__ == "__main__":
    main()