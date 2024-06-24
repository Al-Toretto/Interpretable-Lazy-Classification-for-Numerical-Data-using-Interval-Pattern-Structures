from sklearn.model_selection import GridSearchCV, StratifiedKFold


class HyperparameterTuner:
    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def perform_stratified_grid_search(
        self, X_train, y_train, n_splits=5, scoring="f1", verbose=0, n_jobs=-1
    ):
        stratified_kfolds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=1998
        )
        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=stratified_kfolds,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )
        grid_search.fit(X_train, y_train)
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_estimator_ = grid_search.best_estimator_
