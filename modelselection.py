import numpy as np
import pandas as pd
import pprint
import pickle
from Pipeline import evaluate_model
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from Parametertuning import search_best_SVC, search_best_GBoost, search_best_XGBoost, \
    search_best_LogisticRegression, search_best_KNeighborsClassifier, \
    search_best_AdaBoostClassifier, search_best_MultinomialNB, search_best_ExtraTreesClassifier


X_train, Y_train, test_x, preprocessor = preprocessing()

X_train = preprocessor.fit_transform(X_train)


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms
random_state = 2
# classifiers = []
#
# classifiers.append(XGBClassifier(use_label_encoder=False, random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1))
# classifiers.append(RandomForestClassifier(oob_score=True, random_state=random_state))
# classifiers.append(ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=random_state))
# classifiers.append(GradientBoostingClassifier(random_state=random_state))
# classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression(random_state=random_state))
# classifiers.append(MultinomialNB())
# #classifiers.append(SGDClassifier(random_state=random_state))
# classifiers.append(SVC(probability=True, random_state=random_state))
#
#
#
# cv_results = []
# for classifier in classifiers:
#     cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring='roc_auc', cv=kfold, n_jobs=-1))
#
# cv_means = []
# cv_std = []
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())
# cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["XGBClassifier",
#         "AdaBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "KNeighboors",
#         "LogisticRegression", "MultinomialNB", "SVC"]})
#
# pd.set_option('display.max_rows', None)
# pprint.pprint(cv_res.sort_values(by="CrossValMeans", ascending=False))


# # SVC
# SVMC_best, SVMC_score, SVMC_params = search_best_SVC(X_train, Y_train, kfold, random_state)
# print(SVMC_score, SVMC_params)
# #GB
# GB_best, GB_score, GB_params = search_best_GBoost(X_train, Y_train, kfold, random_state)
# print(GB_score, GB_params)
#XGB
XGB_best, XGB_score, XGB_params = search_best_XGBoost(X_train, Y_train, kfold, random_state)
print(XGB_score, XGB_params)
# #LR
# LR_best, LR_score, LR_params = search_best_LogisticRegression(X_train, Y_train, kfold, random_state)
# print(LR_score, LR_params)
# #KNN
# KNN_best, KNN_score, KNN_params = search_best_KNeighborsClassifier(X_train, Y_train, kfold)
# print(KNN_score, KNN_params)
# #adasboost
# ada_best, ada_score, ada_params = search_best_AdaBoostClassifier(X_train, Y_train, kfold, random_state)
# print(ada_score, ada_params)
# #MLNB
# MLNB_best, MLNB_score, MLNB_params = search_best_MultinomialNB(X_train, Y_train, kfold)
# print(MLNB_score, MLNB_params)
# #MLNB
# ExtC_best, ExtC_score, ExtC_params = search_best_ExtraTreesClassifier(X_train, Y_train, kfold, random_state)
# print(ExtC_score, ExtC_params)


# #save model
# f = open('saved_model/rfc.pickle','wb')
# pickle.dump(rfc,f)
# f.close()
#load model
# f = open('saved_model/rfc.pickle','rb')
# rfc1 = pickle.load(f)
# f.close()
# print(rfc1.predict(X[0:1,:]))
