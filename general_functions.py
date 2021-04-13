"""This module gives the baseline for the very first attempt,
it helps to understand the relationship between data and model.
"""
#  -*- coding: utf-8 -*-
# date: 2021
# author: Jie kai Jia

from collections import Counter
import pickle

import joblib as jl
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


def feature_selection(estimator, feature_names, matrix_x_temp, label_y, th_value):
    """This function select the more important features as training data with LightGBM classifier.

    Args:
        estimator: A tuple that includes model and model name
        feature_names: A list of feature names.
        matrix_x_temp: Training data, input of the model.
        label_y: Output of the model.
        th_value: A threshold value, the function will take the
            features whose importance is higher than th.

    Returns:
        A training data matrix_x after features selection,
            the feature names that are not used and the number of the used feature names.
    """
    # Select from model
    estimator[0].fit(matrix_x_temp, label_y)
    sfm = SelectFromModel(estimator[0], prefit=True, threshold=th_value)
    matrix_x = sfm.transform(matrix_x_temp)

    # how much features whose feature importance is not zero
    feature_score_dict = get_features_score(feature_names, estimator[0].feature_importances_)
    # print feature importance
    txt_name = '{}_feature_importance.txt'.format(estimator[1])
    feature_score_dict_sorted = get_sorted_feature_score(feature_score_dict, txt_name)
    # print selected features
    txt_name = '{}_feature_importance.txt'.format(estimator[1])
    used_feature_name = get_used_feature(matrix_x, feature_score_dict_sorted, txt_name)

    # find non-selected features
    feature_not_used_name = []
    for _, _feature in enumerate(feature_names):
        if feature_names[_] not in used_feature_name:
            feature_not_used_name.append(feature_names[_])

    # create a chromosome like 01011100
    chromosome_temp = []
    feature_name_ivar = feature_names[:-1]
    for i, _ in enumerate(feature_name_ivar):
        if feature_name_ivar[i] in used_feature_name:
            chromosome_temp.append('1')
        else:
            chromosome_temp.append('0')
    chromosome = ''.join(chromosome_temp)
    print('Chromosome:')
    print(chromosome)
    jl.dump(chromosome, './config/{}_chromosome.pkl'.format(estimator[1]))
    print('\n')
    return matrix_x, feature_not_used_name, len(used_feature_name)


def test_data_feature_drop(test_data, feature_name_drop):
    """delete the useless feature"""
    # print feature_name_drop
    for col in feature_name_drop:
        test_data.drop(col, axis=1, inplace=True)
    print("test_data_shape:")
    print(test_data.shape)
    return test_data.as_matrix()


def write_results_to_csv(csv_name, predict_id, predict_list):
    """this function output the prediction as .csv file. """
    result_df = pd.DataFrame()
    result_df['Id'] = predict_id
    result_df['results'] = predict_list

    # output prediction result as a .csv file
    if len(predict_id) == len(predict_list):
        result_df.to_csv('./data/{}'.format(csv_name), index=False)
    else:
        print('ID number and prediction is different.')


def get_features_score(feature_names, feature_importance):
    """This function selects the features whose importance is not zero
    and make a dictionary that contains feature importance scores.
    """
    feature_score_dict = {}

    # get a feature importance dictionary
    for _feature_name, _ in zip(feature_names, feature_importance):
        feature_score_dict[_feature_name] = _
    # calculate the number of nonzero features
    num_zero_features = 0
    for _ in feature_score_dict:
        if feature_score_dict[_] == 0.0:
            num_zero_features += 1
    num_nonzero_features = len(feature_score_dict) - num_zero_features
    print('{0}: {1}'.format('number of not-zero features', num_nonzero_features))

    return feature_score_dict


def get_sorted_feature_score(feature_score_dict, txt_name):
    """This function prints a .txt file and makes a sorted feature score dictionary."""
    feature_score_dict_sorted = sorted(feature_score_dict.items(), key=lambda d: d[1], reverse=True)

    # print sorted feature importance score
    print('feature_importance:')
    for _, _feature in enumerate(feature_score_dict_sorted):
        print(feature_score_dict_sorted[_][0], feature_score_dict_sorted[_][1])
    print('\n')

    # make a file that contains feature importance score
    with open('./feature_analysis/{}'.format(txt_name), 'w', encoding='utf-8') as _f:
        _f.write('Rank\tFeature Name\tFeature Importance\n')
        for _, _feature in enumerate(feature_score_dict_sorted):
            _f.write(str(_) + '\t' + str(feature_score_dict_sorted[_][0]) + '\t' + str(
                feature_score_dict_sorted[_][1]) + '\n')

    return feature_score_dict_sorted


def get_used_feature(matrix_x, feature_score_dict_sorted, txt_name):
    """This function prints a .txt file and returns the used feature name."""
    how_long = matrix_x.shape[1]
    feature_used_dict_temp = feature_score_dict_sorted[:how_long]
    used_feature_name = []

    # print the chosen feature
    for _, _feature in enumerate(feature_used_dict_temp):
        used_feature_name.append(feature_used_dict_temp[_][0])
    print('chosen_feature:')
    for _, _feature in enumerate(used_feature_name):
        print(used_feature_name[_])
    print('\n')

    # make a file that contains the used feature
    with open('./feature_analysis/{}'.format(txt_name), 'w', encoding='utf-8') as _f:
        _f.write('Chosen Feature Name :\n')
        for _, _feature in enumerate(used_feature_name):
            _f.write('{}\n'.format(used_feature_name[_]))

    return used_feature_name


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


def save_model(model, model_name):
    # save model
    with open('./saved_model/{}.pickle'.format(model_name), 'wb') as f:
        pickle.dump(model, f)


def load_model(model_name):
    # load model
    with open('./saved_model/{}.pickle'.format(model_name), 'rb') as f:
        model = pickle.load(f)

    return model
