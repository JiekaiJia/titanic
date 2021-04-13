"""This module provides the functions to tune parameters of different models."""
#  -*- coding: utf-8 -*-
# date: 2021
# author: Jie kai Jia

from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier
)
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def search_best_svc(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of SVC model
    and return the parameters with best results.
    """
    svc = SVC(probability=True, random_state=random_state)
    svc_param_grid = {
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'kernel': ['rbf'],
        'gamma': [0.03, 0.04, 0.05, 0.06, 0.07, 0.1],
        # gamma by default 1/k
        'C': [0.01, 0.1, 0.6, 0.7, 0.8, 0.9, 1]
        # C is penalty coefficient. When c is large,
        # it's easy to over fit, and if c is small, it's easy to under fit.
    }
    svc_model = GridSearchCV(
        svc, param_grid=svc_param_grid,
        cv=k_fold, scoring='neg_log_loss',
        n_jobs=-1, verbose=1
    )
    svc_model.fit(train_x, label_y)

    return svc_model.best_estimator_, svc_model.best_score_, svc_model.best_params_


def search_best_xgb(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of XGBoost model
    and return the parameters with best results.
    """
    xgb = XGBClassifier(use_label_encoder=False, random_state=random_state)
    xgb_param_grid = {
        'learning_rate': [0.1],  # np.linspace(0.05, 0.07, 10)
        'n_estimators': [60],  # range(20, 181, 10)

        'max_depth': [4],  # range(2, 10)
        'min_child_weight': [0.1],
        # np.linspace(0, 0.9, 10) 在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。
        # 因此，某些叶子节点下的值会比较小。
        'gamma': [4.3],
        # 0,by default.np.linspace(0, 9, 10) Minimum loss reduction required to make a
        # further partition on a leaf node of the tree.
        'subsample': [0.79],  # np.linspace(0.5, 1, 20)
        'colsample_bytree': [1],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        # [1e-5, 1e-2, 0.1, 1, 100], L1 regularization term on weights,0
        # 'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100], #L2 regularization term on weights,1
        # 'scale_pos_weight': [default = 1]
        # Control the balance of positive and negative weights, useful for unbalanced classes
    }
    xgb_model = GridSearchCV(
        xgb, param_grid=xgb_param_grid,
        cv=k_fold, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )  # roc_auc,f1,neg_log_loss
    xgb_model.fit(train_x, label_y)

    return xgb_model.best_estimator_, xgb_model.best_score_, xgb_model.best_params_


def search_best_gbm(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of Gradient boost model
    and return the parameters with best results.
    """
    gbm = GradientBoostingClassifier(random_state=random_state)
    gbm_param_grid = {
        'loss': ['deviance'],
        'learning_rate': [0.09],  # np.linspace(0.01, 0.1, 10)
        'n_estimators': [200],  # range(1, 301, 50)

        'max_depth': [3],  # typical range(3, 21, 1)
        'min_samples_split': [0.006666666666666666],
        # 0.5-2% of total observations np.linspace(0.005, 0.02, 10)

        'min_samples_leaf': [3],  # range(1, 10)
        'max_features': [0.5],  # typical sqrt to 30-40% of total features
        'subsample': [0.8],  # [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
    }

    gb_model = GridSearchCV(
        gbm, param_grid=gbm_param_grid,
        cv=k_fold, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    gb_model.fit(train_x, label_y)

    return gb_model.best_estimator_, gb_model.best_score_, gb_model.best_params_


def search_best_lrm(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of logistic regression model
    and return the parameters with best results.
    """
    lrm = LogisticRegression(random_state=random_state)
    lrm_param_grid = {
        'C': [1, 2, 3],  # np.linspace(0.6, 0.8, 10)
        'penalty': ['l1', 'l2'],
        'max_iter': [200],
        # 'class_weight': ['balanced', None],
        # 'solver': ['liblinear']  # ['liblinear', 'sag', 'lbfgs', 'newton-cg', 'saga']
    }

    lr_model = GridSearchCV(
        lrm, param_grid=lrm_param_grid,
        cv=k_fold, scoring='f1',
        n_jobs=-1, verbose=1
    )
    lr_model.fit(train_x, label_y)

    return lr_model.best_estimator_, lr_model.best_score_, lr_model.best_params_


def search_best_knn(train_x, label_y, k_fold):
    """this function is used to tune the parameters of KNN model
    and return the parameters with best results.
    """
    knn = KNeighborsClassifier()
    knn_param_grid = [
        {
            'n_neighbors': [10],
            'weights': ['uniform'],
        },
        {
            'n_neighbors': range(10, 31),
            'weights': ['distance'],
            'p': [1, 2]}
    ]

    knn_model = GridSearchCV(
        knn, param_grid=knn_param_grid,
        cv=k_fold, scoring='f1',
        n_jobs=-1, verbose=1
    )
    knn_model.fit(train_x, label_y)

    return knn_model.best_estimator_, knn_model.best_score_, knn_model.best_params_


def search_best_mnb(train_x, label_y, k_fold):
    """this function is used to tune the parameters of MultinomialNB model
    and return the parameters with best results.
    """
    mnb = MultinomialNB()
    mnb_param_grid = {'alpha': [4.7368]}  # np.linspace(4, 6, 20)

    mnb_model = GridSearchCV(
        mnb, param_grid=mnb_param_grid,
        cv=k_fold, scoring='f1',
        n_jobs=-1, verbose=1
    )
    mnb_model.fit(train_x, label_y)

    return mnb_model.best_estimator_, mnb_model.best_score_, mnb_model.best_params_


def search_best_sgd(train_x, label_y, k_fold):
    """this function is used to tune the parameters of SGDClassifier model
    and return the parameters with best results.
    """
    sgd = SGDClassifier()
    sgd_param_grid = {
        'alpha': [0.01, 0.001, 0.0001],
        'loss': ['hinge', 'log', 'modified_huber']
    }

    sgd_model = GridSearchCV(
        sgd, param_grid=sgd_param_grid,
        cv=k_fold, scoring='f1',
        n_jobs=-1, verbose=1
    )
    sgd_model.fit(train_x, label_y)

    return sgd_model.best_estimator_, sgd_model.best_score_


def search_best_rf(train_x, label_y, k_fold):
    """this function is used to tune the parameters of RandomForest model
    and return the parameters with best results.
    """
    rfm = RandomForestClassifier(oob_score=True)
    rfm_param_grid = {
        'max_depth': range(3, 14, 2),
        'min_samples_split': range(50, 201, 20),
        'min_samples_leaf': range(10, 60, 10)
    }

    rf_model = GridSearchCV(
        rfm, param_grid=rfm_param_grid,
        cv=k_fold, scoring='f1',
        n_jobs=-1, verbose=1
    )
    rf_model.fit(train_x, label_y)

    return rf_model.best_estimator_, rf_model.best_score_


def search_best_dtm(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of DecisionTree model
    and return the parameters with best results.
    """
    dtm = DecisionTreeClassifier(random_state=random_state)
    dtm_param_grid = {
        'max_depth': range(3, 14, 2),
        'min_samples_split': range(50, 201, 20),
        'min_samples_leaf': range(10, 60, 10)
    }

    dt_model = GridSearchCV(
        dtm, param_grid=dtm_param_grid,
        cv=k_fold, scoring='f1',
        n_jobs=-1, verbose=1
    )
    dt_model.fit(train_x, label_y)

    return dt_model.best_estimator_, dt_model.best_score_, dt_model.best_params_


def search_best_ada(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of AdaBoost model
    and return the parameters with best results.
    """
    dtm = DecisionTreeClassifier(random_state=random_state)
    ada_dtm = AdaBoostClassifier(dtm, random_state=random_state)
    ada_param_grid = {
        'base_estimator__max_depth': [6],  # range(2, 14, 1)
        'base_estimator__min_samples_split': [10],  # range(5, 21)
        # 'base_estimator__min_samples_leaf': [1],
        'n_estimators': [40],  # range(30, 70, 10)
        'learning_rate': [0.009778]  # np.linspace(0.009, 0.01, 10)
    }
    ada_model = GridSearchCV(
        ada_dtm, param_grid=ada_param_grid,
        cv=k_fold, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    ada_model.fit(train_x, label_y)

    return ada_model.best_estimator_, ada_model.best_score_, ada_model.best_params_


def search_best_etm(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of ExtraTree model
    and return the parameters with best results.
    """
    # ExtraTrees
    etm = ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=random_state)
    # Search grid for optimal parameters
    etm_param_grid = {
        'n_estimators': [30],  # range(50, 250, 50)
        'max_depth': [8],  # range(2, 15)
        'min_samples_split': [3],  # range(2, 21)
        # 'min_samples_leaf': range(1, 10),
        'max_features': [9]  # range(2, 21)
        }
    et_model = GridSearchCV(
        etm, param_grid=etm_param_grid,
        cv=k_fold, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    et_model.fit(train_x, label_y)

    return et_model.best_estimator_, et_model.best_score_, et_model.best_params_


def search_best_lgb(train_x, label_y, k_fold, random_state):
    """this function is used to tune the parameters of LightGB model
    and return the parameters with best results.
    """
    lgb = LGBMClassifier(random_state=random_state)

    # hyper_parameter optimization
    lgb_param_grid = {
        'max_depth': 6,
        'num_leaves': 64,
        'learning_rate': 0.03,
        'scale_pos_weight': 1,
        'num_threads': 40,
        'objective': 'binary',
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_sum_hessian_in_leaf': 100
        # param['is_unbalance'] = 'true'
        # param['metric'] = 'auc'
    }

    lgb_model = GridSearchCV(
        lgb, param_grid=lgb_param_grid,
        cv=k_fold, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    lgb_model.fit(train_x, label_y)

    return lgb_model.best_estimator_, lgb_model.best_score_, lgb_model.best_params_


def raw_model_compare(train_x, label_y):
    """This function is used to compare model without parameter tuning."""
    # Cross validate model with K_fold stratified cross val
    k_fold = StratifiedKFold(n_splits=10)
    # Modeling step Test different algorithms
    random_state = 2021
    classifiers = [
        AdaBoostClassifier(
            DecisionTreeClassifier(random_state=random_state),
            random_state=random_state,
            learning_rate=0.1
        ),
        ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        KNeighborsClassifier(),
        LGBMClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
        MultinomialNB(),
        RandomForestClassifier(oob_score=True, random_state=random_state),
        SGDClassifier(random_state=random_state),
        SVC(probability=True, random_state=random_state),
        XGBClassifier(use_label_encoder=False, random_state=random_state)
        ]
    cv_results = []
    for classifier in classifiers:
        cv_results.append(
            cross_val_score(
                classifier, train_x,
                y=label_y, scoring='accuracy',
                cv=k_fold, n_jobs=-1
            )
        )

    cv_means = []
    cv_stds = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_stds.append(cv_result.std())
    # put the results in a dictionary
    cv_dict = {
        'CrossValMeans': cv_means,
        'CrossValErrors': cv_stds,
        'Algorithm': [
            'AdaBoost', 'ExtraTrees', 'GradientBoost', 'KNN',
            'LGBoost', 'LogisticRegression', 'MultinomialNB',
            'RandomForest', 'SGD', 'SVM', 'XGBoost'
        ]
    }
    cv_res = pd.DataFrame(cv_dict)

    # generate a .txt file with cross validation results
    with open('./results/compare_without_tuning.txt', 'w') as _f:
        _f.write('Algorithm\tCrossValMeans\tCrossValErrors\n')
        for cv_mean, cv_std, algorithm in zip(cv_means, cv_stds, cv_dict['Algorithm']):
            _f.write('{}\t{}\t{}\n'.format(algorithm, cv_mean, cv_std))

    return cv_res

    # （1）num_leaves
    #
    # LightGBM使用的是leaf - wise的算法，因此在调节树的复杂程度时，使用的是num_leaves而不是max_depth。
    #
    # 大致换算关系：num_leaves = 2 ^ (max_depth)
    #
    # （2）样本分布非平衡数据集：可以param[‘is_unbalance’]=’true’
    #
    # （3）Bagging参数：bagging_fraction + bagging_freq（必须同时设置）、feature_fraction
    #
    # （4）min_data_in_leaf、min_sum_hessian_in_leaf
    # 选择一个相对来说稍微高一点的learning rate。一般默认的值是0.1，不过针对不同的问题，0.05到0.2之间都可以
    # 决定当前learning rate下最优的决定树数量。它的值应该在40-70之间。记得选择一个你的电脑还能快速运行的值，因为之后这些树会用来做很多测试和调参。
    # 接着调节树参数来调整learning rate和树的数量。我们可以选择不同的参数来定义一个决定树，后面会有这方面的例子
    # 降低learning rate，同时会增加相应的决定树数量使得模型更加稳健
