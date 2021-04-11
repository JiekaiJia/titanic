"""This module gives the baseline for the very first attempt,
it helps to understand the relationship between data and model.
"""
#  -*- coding: utf-8 -*-
# date: 2021
# author: Jiekai Jia

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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def search_best_SVC(x, y, kfold, random_state):
    # SVC classifier
    # C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
    # gamma设的太大，方差会很小，方差很小的高斯分布长得又高又瘦， 会造成只会作用于支持向量样本附近，对于未知样本分类效果很差,默认值是1/k（k是特征数）
    SVMC = SVC(probability=True, random_state=random_state)
    svc_param_grid = {
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'kernel': ['rbf'],
        'gamma': [0.03, 0.04, 0.05, 0.06, 0.07, 0.1],
        'C': [0.01, 0.1, 0.6, 0.7, 0.8, 0.9, 1]
    }
    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring='neg_log_loss', n_jobs=-1, verbose=1)
    gsSVMC.fit(x, y)
    return gsSVMC.best_estimator_, gsSVMC.best_score_, gsSVMC.best_params_


def search_best_XGBoost(x, y, kfold, random_state):
    # Cross validate model with Kfold stratified cross val
    XGBC = XGBClassifier(use_label_encoder=False, random_state=random_state)
    XGBC_param_grid = {
        "learning_rate": [0.1],  # np.linspace(0.05, 0.07, 10)
        'n_estimators': [60],  # range(20, 181, 10)

        "max_depth": [4],  # range(2, 10)
        "min_child_weight": [0.1],  # np.linspace(0, 0.9, 10) 在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。
                                    # 因此，某些叶子节点下的值会比较小。
        'gamma': [4.3],  # 0,bydefault.np.linspace(0, 9, 10) Minimum loss reduction required to make a
                         # further partition on a leaf node of the tree.
        "subsample": [0.79],  # np.linspace(0.5, 1, 20)
        "colsample_bytree": [1],
        # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100], #[1e-5, 1e-2, 0.1, 1, 100], L1 regularization term on weights,0
        # 'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100], #L2 regularization term on weights,1
        # 'scale_pos_weight': [default = 1] Control the balance of positive and negative weights,
        # useful for unbalanced classes
    }
    gsXGBC = GridSearchCV(XGBC, param_grid=XGBC_param_grid, cv=kfold, scoring='roc_auc', n_jobs=-1, verbose=1)# roc_auc,f1,neg_log_loss
    gsXGBC.fit(x, y)
    return gsXGBC.best_estimator_, gsXGBC.best_score_, gsXGBC.best_params_


def search_best_GBoost(x, y, kfold, random_state):
    # 选择一个相对来说稍微高一点的learning rate。一般默认的值是0.1，不过针对不同的问题，0.05到0.2之间都可以
    # 决定当前learning rate下最优的决定树数量。它的值应该在40-70之间。记得选择一个你的电脑还能快速运行的值，因为之后这些树会用来做很多测试和调参。
    # 接着调节树参数来调整learning rate和树的数量。我们可以选择不同的参数来定义一个决定树，后面会有这方面的例子
    # 降低learning rate，同时会增加相应的决定树数量使得模型更加稳健
    GBC = GradientBoostingClassifier(random_state=random_state)
    GBC_param_grid = {
        'loss': ["deviance"],
        'learning_rate': [0.09],  # np.linspace(0.01, 0.1, 10)
        'n_estimators': [200],  # range(1, 301, 50)

        'max_depth': [3],  # typical range(3, 21, 1)
        'min_samples_split': [0.006666666666666666],  # 0.5-2% of total observations np.linspace(0.005, 0.02, 10)

        'min_samples_leaf': [3],  # range(1, 10)
        'max_features': [0.5],  # typical sqrt to 30-40% of total features
        'subsample': [0.8],  # [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
    }

    gsGBC = GridSearchCV(GBC, param_grid=GBC_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose=1)
    gsGBC.fit(x, y)
    return gsGBC.best_estimator_, gsGBC.best_score_, gsGBC.best_params_


def search_best_LogisticRegression(x, y, kfold, random_state):
    LR = LogisticRegression(random_state=random_state)
    LR_param_grid = {
        'C': [1, 2, 3],  # np.linspace(0.6, 0.8, 10)
        'penalty': ['l1', 'l2'],
        'max_iter': [200],
        # 'class_weight': ['balanced', None],
        # 'solver': ['liblinear'] #['liblinear', 'sag', 'lbfgs', 'newton-cg', 'saga']
    }

    gsLR = GridSearchCV(LR, param_grid=LR_param_grid, cv=kfold, scoring="f1", n_jobs=-1, verbose=1)
    gsLR.fit(x, y)
    return gsLR.best_estimator_, gsLR.best_score_, gsLR.best_params_


def search_best_KNeighborsClassifier(x, y, kfold):
    KNN = KNeighborsClassifier()
    KNN_param_grid = [
        {
            'n_neighbors': [10],
            'weights': ["uniform"],
        },
        {
            'n_neighbors': range(10, 31),
            'weights': ["distance"],
            'p': [1, 2]}
    ]

    gsKNN = GridSearchCV(KNN, param_grid=KNN_param_grid, cv=kfold, scoring="f1", n_jobs=-1, verbose=1)
    gsKNN.fit(x, y)
    return gsKNN.best_estimator_, gsKNN.best_score_, gsKNN.best_params_


def search_best_MultinomialNB(x, y, kfold):
    MNB = MultinomialNB()
    MNB_param_grid = {'alpha': [4.7368]}  # np.linspace(4, 6, 20)

    gsMNB = GridSearchCV(MNB, param_grid=MNB_param_grid, cv=kfold, scoring="f1", n_jobs=-1, verbose=1)
    gsMNB.fit(x, y)
    return gsMNB.best_estimator_, gsMNB.best_score_, gsMNB.best_params_


def search_best_SGDClassifier(x, y, kfold):
    SGDC = SGDClassifier()
    SGDC_param_grid = {
        'alpha': [0.01, 0.001, 0.0001],
        'loss': ["hinge", "log", "modified_huber"]
    }

    gsSGDC = GridSearchCV(SGDC, param_grid=SGDC_param_grid, cv=kfold, scoring="f1", n_jobs=-1, verbose=1)
    gsSGDC.fit(x, y)
    return gsSGDC.best_estimator_, gsSGDC.best_score_


def search_best_RandomForestClassifier(x, y, kfold):
    RFC = RandomForestClassifier(oob_score=True)
    RFC_param_grid = {
        "max_depth": range(3, 14, 2),
        'min_samples_split': range(50, 201, 20),
        'min_samples_leaf': range(10, 60, 10)
    }

    gsRFC = GridSearchCV(RFC, param_grid=RFC_param_grid, cv=kfold, scoring="f1", n_jobs=-1, verbose=1)
    gsRFC.fit(x, y)
    return gsRFC.best_estimator_, gsRFC.best_score_


def search_best_DecisionTreeClassifier(x, y, kfold, random_state):
    DT = DecisionTreeClassifier(random_state=random_state)
    DT_param_grid = {
        "max_depth": range(3, 14, 2),
        'min_samples_split': range(50, 201, 20),
        'min_samples_leaf': range(10, 60, 10)
    }

    gsDT = GridSearchCV(DT, param_grid=DT_param_grid, cv=kfold, scoring="f1", n_jobs=-1, verbose=1)
    gsDT.fit(x, y)
    return gsDT.best_estimator_, gsDT.best_score_, gsDT.best_params_


def search_best_AdaBoostClassifier(x, y, kfold, random_state):
    DTC = DecisionTreeClassifier(random_state=random_state)
    adaDTC = AdaBoostClassifier(DTC, random_state=random_state)
    ada_param_grid = {
        "base_estimator__max_depth": [6],  # range(2, 14, 1)
        'base_estimator__min_samples_split': [10],  # range(5, 21)
        # 'base_estimator__min_samples_leaf': [1],
        "n_estimators": [40],  # range(30, 70, 10)
        "learning_rate": [0.009778]  # np.linspace(0.009, 0.01, 10)
    }
    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsadaDTC.fit(x, y)
    return gsadaDTC.best_estimator_, gsadaDTC.best_score_, gsadaDTC.best_params_


def search_best_ExtraTreesClassifier(x, y, kfold, random_state):
    # ExtraTrees
    ExtC = ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=random_state)
    # Search grid for optimal parameters
    ExtC_param_grid = {
        "n_estimators": [30],  # range(50, 250, 50)
        "max_depth": [8],  # range(2, 15)
        "min_samples_split": [3],  # range(2, 21)
        # "min_samples_leaf": range(1, 10),
        "max_features": [9]  # range(2, 21)
        }
    gsExtC = GridSearchCV(ExtC, param_grid=ExtC_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsExtC.fit(x, y)
    return gsExtC.best_estimator_, gsExtC.best_score_, gsExtC.best_params_
