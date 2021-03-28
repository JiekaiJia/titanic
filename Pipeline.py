from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def evaluate_model(train_x, train_y, preprocessor, my_model, param_grid):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', my_model)
                                  ])

    searchCV = GridSearchCV(my_pipeline, cv=5, scoring='f1', param_grid=param_grid, n_jobs=-1)
    searchCV.fit(train_x, train_y)

    best_params = searchCV.best_params_
    best_score = searchCV.best_score_
    best_estimator = searchCV.best_estimator_

    return best_score, best_params, best_estimator