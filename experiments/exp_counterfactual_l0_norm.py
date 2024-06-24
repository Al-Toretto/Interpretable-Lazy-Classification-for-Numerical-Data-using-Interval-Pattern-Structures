from src.dataset import Dataset, known_datasets
from src.dataset_preprocessor import DatasetPreprocessor
from src.ips_knn_classifier import IPSKNNClassifier
from src.eager_ips_knn_classifier import EagerIPSKNNClassifier
import pandas as pd
from config.optimal_hyperparameters import optimal_params
from src.utils import custom_print



def perform_reduction_of_number_of_needed_features_analysis(df1 : pd.DataFrame, df2 : pd.DataFrame, mask: pd.Series, filename: str):
    df1_reduced = df1[mask].drop('l1_dist', axis=1)
    df2_reduced = df2[mask].drop('l1_dist', axis=1)

    count_non_zeros_1 = (df1_reduced != 0).sum(axis=1)
    count_non_zeros_2 = (df2_reduced != 0).sum(axis=1)

    reduced_non_zero_count = count_non_zeros_1 - count_non_zeros_2
    reduction_ratio = reduced_non_zero_count / count_non_zeros_1
    positive_reduction = reduction_ratio[reduction_ratio > 0]
    negative_reduction = reduction_ratio[reduction_ratio < 0]

    positive_stats = {
        'ratio': len(positive_reduction) / len(reduction_ratio),
        'avg': positive_reduction.mean(),
        'std': positive_reduction.std(),
        'min': positive_reduction.min(),
        'max': positive_reduction.max(),
    }

    negative_stats = {
        'ratio': len(negative_reduction) / len(reduction_ratio),
        'avg': negative_reduction.mean(),
        'std': negative_reduction.std(),
        'min': negative_reduction.min(),
        'max': negative_reduction.max()
    }
    custom_print(f"Positive Reduction Stats: {positive_stats}", filename)
    custom_print(f"Negative Reduction Stats: {negative_stats}", filename)


def main():
    filename = 'exp_counterfactual_l0_norm.txt'
    for name in known_datasets:
        custom_print(f"Dataset ------------ {name} ------------------------------------", filename)
        custom_print("------------------------------------------------------------------------------", filename)
        dataset = Dataset(dataset_name=name)
        preprocessor = DatasetPreprocessor(dataset).preprocess().standardize()

        # IPS-KNN === k-NN-----------
        classifier_ips_knn = IPSKNNClassifier(**optimal_params['ips_knn'][name])
        classifier_ips_knn.fit(dataset.X_train, dataset.y_train)
        classifier_ips_knn.destandardize_features(preprocessor)
        # E-IPS-KNN-T
        classifier_e_ips_knn_t = EagerIPSKNNClassifier(**optimal_params['e_ips_knn_t'][name])
        classifier_e_ips_knn_t.fit(dataset.X_train, dataset.y_train)
        classifier_e_ips_knn_t.destandardize_hyperrectangles(preprocessor)
        # E-IPS-KNN-T&F
        classifier_e_ips_knn_tf = EagerIPSKNNClassifier(**optimal_params['e_ips_knn_tf'][name])
        classifier_e_ips_knn_tf.fit(dataset.X_train, dataset.y_train)
        classifier_e_ips_knn_tf.destandardize_hyperrectangles(preprocessor)
        print('Done Fitting')

        def count_zeros(row):
            return (row==0).sum()
        
        # IPS-KNN === k-NN
        base_y_pred, _, _, _, _, cf_base, _ =  classifier_ips_knn.predict_with_explanation(dataset.X_test, False, False, True, True, False)
        cf_base = cf_base.loc[cf_base.groupby(level='index').apply(lambda x : x.apply(count_zeros, axis=1).idxmax())].droplevel(level='counterfactual')
        
        # E-IPS-KNN-T
        custom_print('E-IPS-KNN T: ******************************************************************', filename)
        t_y_pred, _, _, _, _, cf_t = classifier_e_ips_knn_t.predict_with_explanation(dataset.X_test, False, False, True, True)
        cf_t = cf_t.loc[cf_t.groupby(level='index').apply(lambda x : x.apply(count_zeros, axis=1).idxmax())].droplevel(level='id')
        mask = (base_y_pred == t_y_pred).sort_index()
        perform_reduction_of_number_of_needed_features_analysis(cf_base, cf_t, mask, filename)

        # E-IPS-KNN-T&F
        custom_print('E-IPS-KNN T&F: ****************************************************************', filename)
        tf_y_pred, _, _, _, _, cf_tf = classifier_e_ips_knn_tf.predict_with_explanation(dataset.X_test, False, False, True, True)
        cf_tf = cf_tf.loc[cf_tf.groupby(level='index').apply(lambda x : x.apply(count_zeros, axis=1).idxmax())].droplevel(level='id')
        mask = (base_y_pred == tf_y_pred).sort_index()
        perform_reduction_of_number_of_needed_features_analysis(cf_base, cf_tf, mask, filename)


if __name__ == "__main__":
    main()