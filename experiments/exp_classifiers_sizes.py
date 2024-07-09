from src.dataset import Dataset, known_datasets
from src.dataset_preprocessor import DatasetPreprocessor
from src.ips_knn_classifier import IPSKNNClassifier
from src.utils import custom_print
from config.optimal_hyperparameters import optimal_params


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np



def main():
    filename = "exp_classifiers_sizes.txt"
    for name in known_datasets:
        custom_print(f"Dataset ------------ {name} ------------------------------------", filename)
        custom_print("------------------------------------------------------------------------------", filename)
        dataset = Dataset(dataset_name=name)
        preprocessor = DatasetPreprocessor(dataset).preprocess()

        n_features = len(dataset.X.columns)
        # Naive Bayes: ******************************************************************
        custom_print('Naive Bayes: ******************************************************************', filename)
        custom_print(f'size: {4 * n_features + 2}', filename)
        custom_print("\n\n", filename)


        # XGBoost: **********************************************************************
        custom_print('XGBoost: **********************************************************************', filename)
        custom_print(f"n_estimators = {optimal_params['xgboost'][name]['n_estimators']}", filename)
        custom_print(f"max_depth = {optimal_params['xgboost'][name]['max_depth']}", filename)
        custom_print("\n\n", filename)

        preprocessor = preprocessor.standardize()

        # IPS-KNN: **********************************************************************
        custom_print('IPS-KNN: **********************************************************************', filename)
        classifier = IPSKNNClassifier(**optimal_params['ips_knn'][name])
        classifier.fit(dataset.X_train, dataset.y_train)
        classifier.destandardize_features(preprocessor)
        _, _, size_dict, _, _ = classifier._predict_with_explanation(dataset.X_test, True, False)
        size_list = list(size_dict.values())
        custom_print(size_list, filename)
        custom_print(f"max size = {np.max(size_list)}", filename)
        custom_print(f"avg size = {np.average(size_list)}", filename)
        custom_print("\n\n", filename)
        # KNN: **************************************************************************
        custom_print('KNN: **************************************************************************', filename)
        custom_print(f"size: {n_features * optimal_params['knn'][name]['n_neighbors']}", filename)
        custom_print("\n\n", filename)

        # Logistic Regression: **********************************************************
        custom_print('Logistic Regression: **********************************************************', filename)
        custom_print(f'size: {n_features + 1}', filename)
        custom_print("\n\n", filename)

        # SVM: **************************************************************************
        custom_print('SVM: **************************************************************************', filename)
        if optimal_params['svm'][name]['kernel'] == 'linear':
            custom_print(f'size: {n_features + 1}', filename)
        else:
            classifier = SVC(random_state=1998, **optimal_params['svm'][name])
            classifier.fit(dataset.X_train, dataset.y_train)
            n_support_vectors = np.sum(classifier.n_support_)
            custom_print(f"num_supp_vecs = {n_support_vectors}", filename)
            custom_print(f"size: {n_features * n_support_vectors + n_support_vectors + 2}", filename)
        custom_print("\n\n", filename)

        # Decision Tree: ****************************************************************
        custom_print('Decision Tree: ****************************************************************', filename)
        classifier = DecisionTreeClassifier(random_state=1998, **optimal_params['decision_tree'][name])
        classifier.fit(dataset.X_train, dataset.y_train)
        custom_print(f"max_depth: {optimal_params['decision_tree'][name]['max_depth']}", filename)
        decision_paths = classifier.decision_path(dataset.X_test)

        n_features_per_sample = []
        for row in range(decision_paths.shape[0]):
            sample_node_indices = decision_paths.indices[decision_paths.indptr[row]:decision_paths.indptr[row+1]]
            features_used = classifier.tree_.feature[sample_node_indices]
            unique_features = np.unique(features_used[features_used >= 0])
            n_features_per_sample.append(len(unique_features))
        custom_print(f"max_depth: {np.max(n_features_per_sample)}", filename)
        custom_print(f"avg_depth: {np.average(n_features_per_sample)}", filename)
        custom_print("\n\n", filename)

        # Random Forest: ****************************************************************
        custom_print('Random Forest: ****************************************************************', filename)
        custom_print(f"n_estimators: {optimal_params['random_forest'][name]['n_estimators']}", filename)
        if optimal_params['random_forest'][name]['max_depth'] is not None:
            custom_print(f"max_depth: {optimal_params['random_forest'][name]['max_depth']}", filename)
        else:
            classifier = RandomForestClassifier(random_state=1998, n_jobs=1, **optimal_params['random_forest'][name])
            classifier.fit(dataset.X_train, dataset.y_train)
            max_depths = [estimator.tree_.max_depth for estimator in classifier.estimators_]
            custom_print(f"max_depth = {max(max_depths)}", filename)
        custom_print("\n\n", filename)





if __name__ == "__main__":
    main()
