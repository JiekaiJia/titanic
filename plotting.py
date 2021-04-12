"""This module provides 2 plotting functions, learning curve and ROC curve."""
#  -*- coding: utf-8 -*-
# date: 2021
# author: Jie kai Jia

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    auc,
    f1_score,
    roc_curve
)


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


def plot_roc_curve(estimator, cross_validation, train_x, label_y, th_value):
    """
    Args:
        estimator:
        cross_validation:
        train_x:
        label_y:
        th_value:
    """
    # Model fit, predict and ROC
    colors = cycle(['cyan', 'indigo', 'sea_green', 'orange', 'blue'])
    line_width = 2
    mean_f1 = 0.0
    mean_tpr = []
    mean_fpr = np.linspace(0, 1, 500)
    i_of_roc = 0
    cv_split = cross_validation.split(train_x, label_y)

    for (train_indices, test_indices), color in zip(cv_split, colors):
        a_model = estimator.fit(train_x[train_indices], label_y[train_indices])
        # y_predict_label = a_model.predict(x[test_indices])
        predict_pr = a_model.predict_proba(train_x[test_indices])
        fpr, tpr, _ = roc_curve(label_y[test_indices], predict_pr[:, 1])

        mean_tpr.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=line_width, color=color,
                 label='ROC fold %d (area = %0.4f)' % (i_of_roc, roc_auc))
        i_of_roc += 1

        label_transformed = predict_pr[:, 1]
        for i, _ in enumerate(label_transformed):
            if label_transformed[i] > th_value:
                label_transformed[i] = 1
            else:
                label_transformed[i] = 0
        f1_score_ = f1_score(label_y[test_indices], label_transformed.astype('int64'))
        mean_f1 += f1_score_

    plt.plot([0, 1], [0, 1], linestyle='--', lw=line_width, color='k', label='Luck')

    mean_tpr = np.mean(mean_tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('mean_auc={}'.format(mean_auc))
    print('mean_f1={}'.format(mean_f1 / 5))
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.4f)' % mean_auc, lw=line_width)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate mean_f1:{}'.format(mean_f1 / 5))
    plt.ylabel('True Positive Rate')

    plt.title('ROC_gbm{}_features_f1_{}'.format(train_x.shape(1), mean_f1 / 5))
    plt.legend(loc="lower right")
    plt.savefig(
        '''./figure/predict_ROC_XL_N_{0}_features_
        {1}_pr_to_label_using_th_{2}.png'''.format(100, train_x.shape(1), th_value))
    plt.show()
