#!/usr/bin/env python3.8
#  -*- coding: utf-8 -*-
# date: 2021
# author: Jiekai Jia

"""this module provides some basic functions to help analyze data."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def detect_outliers(dataframe: pd.DataFrame, n_outlier: int, features: list) -> list:
    """Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        value_q1 = np.percentile(dataframe[col], 25)
        # 3rd quartile (75%)
        value_q3 = np.percentile(dataframe[col], 75)
        # Interquartile range (IQR)
        iqr = value_q3 - value_q1
        # outlier step
        outlier_step = 1.5 * iqr
        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataframe[(dataframe[col] < value_q1 - outlier_step)
                                     | (dataframe[col] > value_q3 + outlier_step)].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(_ for _ in outlier_indices if outlier_indices[_] > n_outlier)

    return multiple_outliers


def plot_learning_curve(estimator: object, title: str, x_train: pd.DataFrame,
                        label_y: pd.Series, cross_validation: iter = None):
    """Generate a simple plot of the validation and training learning curve."""
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, x_train, label_y, cv=cross_validation, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.grid()
    # plot learning curve of training set
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    # plot learning curve of validation set
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


def get_preprocessor_by(num_features: list, cat_features: list) -> object:
    """get a preprocessing pipeline with the given numeric and categorical features."""
    # a pipeline that imputes and scales numeric features
    numerical_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="mean")),
                                            ('scale', StandardScaler())
                                            ])
    # a pipeline that imputes categorical features and transforms features into one_shot code
    categorical_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                              ("one_hot", OneHotEncoder(handle_unknown="ignore"))
                                              ])
    # a transformer that combines numerical and categorical pipeline
    preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, num_features),
                                                   ("cat", categorical_transformer, cat_features)
                                                   ])

    return preprocessor


def get_best_model(x_train: pd.DataFrame, label_y: pd.Series,
                   preprocessor: object, model: object, param_grid: dict) -> object:
    """train model with cross validation and choose the best one."""
    # create a model training pipeline with a preprocessor and the chosen model
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])
    # train model with GridsearchCV
    my_model = GridSearchCV(my_pipeline, cv=5, scoring='f1', param_grid=param_grid, n_jobs=-1)
    my_model.fit(x_train, label_y)
    # get the best model, score and parameters
    best_params = my_model.best_params_
    best_score = my_model.best_score_
    best_estimator = my_model.best_estimator_

    return best_score, best_params, best_estimator
