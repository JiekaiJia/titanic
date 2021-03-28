import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import matplotlib.pyplot as plt
import pprint

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from Parametertuning import search_best_SVC, search_best_GBoost, search_best_XGBoost, \
    search_best_LogisticRegression, search_best_KNeighborsClassifier, \
    search_best_AdaBoostClassifier, search_best_MultinomialNB, search_best_ExtraTreesClassifier
from Evaluation import plot_learning_curve

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
IDtest = test_data['PassengerId']

#train_data.info()
#print("_" * 40)
#test_data.info()
# train_data.describe(include=['O'])

# # detect outliers from Age, SibSp , Parch and Fare
# Outliers_to_drop = detect_outliers(train_data, 2, ["Age", "SibSp", "Parch", "Fare"])
#
# # Drop outliers
# train_data = train_data.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(train_data)
dataset = pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)

#Data washing
#fill Cabin
dataset['Cabin'] = dataset['Cabin'].fillna('U')
#print(dataset['Cabin'].head())
# Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode().iloc[0])

# Fill Fare missing values with the median value
# dataset["Fare"] = dataset.groupby(["Embarked", "Pclass"], as_index=False)["Fare"].transform(
#     lambda x: x.fillna(x.mean()))
# Apply log to Fare to reduce skewness distribution
dataset['Fare'] = dataset['Fare'].map(lambda x: np.log(x) if x > 0 else 0)

dataset['Fare'] = dataset['Fare'].fillna(dataset[(dataset['Pclass'] == 3) & (dataset['Embarked'] == 'S') & (dataset['Cabin'] == 'U')]['Fare'].mean())
#feature engineering
# explore Name feature
# for data in combine_dropdata:
# data["Title"] = data.Name.str.extract("([A-Za-z]+)\.", expand=False)

# Get Title from Name
dataset['Title'] = dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
dataset['Title'] = dataset['Title'].replace(['the Countess', 'Don', 'Lady', 'Sir', 'Dona'], 'Royalty')
dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace(['Mme', 'Ms'], 'Mrs')
dataset['Title'] = dataset['Title'].replace('Jonkheer',  'Master')
#print(dataset['Title'].value_counts())
# convert Sex into categorical value 0 for male and 1 for female
#dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})

# convert Title into categorical value
#dataset["Title"] = dataset["Title"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Aristocrat": 4, "Rev": 5})
# Filling missing value of Age
#subdata = dataset[dataset["Age"].notnull()]
#sub_x = subdata[["SibSp", "Parch", "Pclass", "Title"]]
#sub_y = subdata["Age"]
#x_subtrain, x_subtest, y_subtrain, y_subtest = train_test_split(sub_x, sub_y, test_size=0.2, random_state=0)
#Age_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False)
#Age_model.fit(x_subtrain,y_subtrain)
#print(mean_squared_error(y_subtest, Age_model.predict(x_subtest)))
# Index of NaN age rows
# index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
#
# for i in index_NaN_age:
#     dataset.loc[:, ['Age']].iloc[i] = Age_model.predict(dataset[["SibSp", "Parch", "Pclass", "Title"]])[i]
# dataset.loc[dataset["Age"] < 15, "Age"] = 0
# dataset.loc[(dataset["Age"] < 55) & (dataset["Age"] >= 15), "Age"] = 1
# dataset.loc[dataset["Age"] >= 55, "Age"] = 2
# dataset["Age"] = dataset["Age"].astype("category")
# dataset = pd.get_dummies(dataset, columns = ["Age"], prefix="A")
dataset['FamilyNum'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset.loc[dataset['FamilyNum'] == 1, 'FamilySize'] = 0
dataset.loc[(dataset['FamilyNum'] > 1) & (dataset['FamilyNum'] <= 4), 'FamilySize'] = 1
dataset.loc[dataset['FamilyNum'] > 4, 'FamilySize'] = 2
#print(dataset['Familysize'].value_counts())
#dataset['Familysize'] = dataset['Familysize'].astype("category")
#dataset = pd.get_dummies(dataset, columns=["Familysize"], prefix="F")
# convert to indicator values Embarked
#dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
dataset['Deck'] = dataset['Cabin'].map(lambda x: x[0])
# sns.barplot(data=dataset, x='Deck', y='Survived')
# plt.show()
# Ticket = []
# for i in list(dataset.Ticket):
#     if not i.isdigit():
#         Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
#     else:
#         Ticket.append("X")
#
# dataset["Ticket"] = Ticket
# dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
TickCountDict = dataset['Ticket'].value_counts()
#print(TickCountDict.head())
dataset['TickCot'] = dataset['Ticket'].map(TickCountDict)
#print(dataset['Tickcot'].head)
dataset.loc[(dataset['TickCot'] >= 2) & (dataset['TickCot'] <= 4), 'TickGroup'] = 0
dataset.loc[dataset['TickCot'] == 1, 'TickGroup'] = 1
dataset.loc[(dataset['TickCot'] >= 5) & (dataset['TickCot'] <= 8), 'TickGroup'] = 1
dataset.loc[dataset['TickCot'] > 8, 'TickGroup'] = 2
#sns.barplot(data=dataset, x='TickGroup', y='Survived')
#plt.show()
AgePre = dataset[['Age', 'Parch', 'Pclass', 'SibSp', 'Title', 'FamilyNum', 'TickCot']]
AgePre = pd.get_dummies(AgePre)
ParAge = pd.get_dummies(AgePre['Parch'], prefix='Parch')
SibAge = pd.get_dummies(AgePre['SibSp'], prefix='SibSp')
PclAge = pd.get_dummies(AgePre['Pclass'], prefix='Pclass')
AgeCorrDf = AgePre.corr()
#print(AgeCorrDf['Age'].sort_values())
AgePre = pd.concat([AgePre, ParAge, SibAge, PclAge], axis=1)
#print(AgePre.head())
AgeKnown = AgePre[AgePre['Age'].notnull()]
AgeUnKnown = AgePre[AgePre['Age'].isnull()]
AgeKnown_X = AgeKnown.drop(['Age'], axis=1)
AgeKnown_y = AgeKnown['Age']
AgeUnKnown_X = AgeUnKnown.drop(['Age'], axis=1)
rfr = RandomForestRegressor(random_state=None, n_estimators=500, n_jobs=-1)
rfr.fit(AgeKnown_X, AgeKnown_y)
#print(rfr.score(AgeKnown_X, AgeKnown_y))
AgeUnKnown_y = rfr.predict(AgeUnKnown_X)
dataset.loc[dataset['Age'].isnull(), ['Age']] = AgeUnKnown_y
#print(dataset.info())
#同组识别
dataset['Surname'] = dataset['Name'].map(lambda x: x.split(',')[0].strip())
SurNameDict = dataset['Surname'].value_counts()
dataset['SurnameNum'] = dataset['Surname'].map(SurNameDict)
MaleDf = dataset[(dataset['Sex'] == 'male') & (dataset['Age'] > 12) & (dataset['FamilyNum'] >= 2)]
FemChildDf = dataset[((dataset['Sex'] == 'female') | (dataset['Age'] <= 12)) & (dataset['FamilyNum'] >= 2)]
MSurNamDf = MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
MSurNamDict = MSurNamDf[MSurNamDf.values == 1].index
#print(MSurNamDict)
FCSurNamDf = FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
#FCSurNamDf.head()
#print(FCSurNamDf.value_counts())
#print(MSurNamDf.value_counts())
FCSurNamDict = FCSurNamDf[FCSurNamDf.values == 0].index
#print(FCSurNamDict)
#对数据集中这些姓氏的男性数据进行修正：1、性别改为女；2、年龄改为5。
dataset.loc[(dataset['Survived'].isnull()) & (dataset['Surname'].isin(MSurNamDict)) & (dataset['Sex'] == 'male'), 'Age'] = 5
dataset.loc[(dataset['Survived'].isnull()) & (dataset['Surname'].isin(MSurNamDict)) & (dataset['Sex'] == 'male'), 'Sex'] = 'female'

#对数据集中这些姓氏的女性及儿童的数据进行修正：1、性别改为男；2、年龄改为60。
dataset.loc[(dataset['Survived'].isnull()) & (dataset['Surname'].isin(FCSurNamDict)) & ((dataset['Sex'] == 'female') | (dataset['Age'] <= 12)), 'Age'] = 60
dataset.loc[(dataset['Survived'].isnull()) & (dataset['Surname'].isin(FCSurNamDict)) & ((dataset['Sex'] == 'female') | (dataset['Age'] <= 12)), 'Sex'] = 'male'
#人工筛选
fullSel = dataset.drop(['Cabin', 'Name', 'Ticket', 'PassengerId', 'Surname', 'SurnameNum'], axis=1)
#查看各特征与标签的相关性
corrDf = fullSel.corr()
#print(corrDf['Survived'].sort_values(ascending=True))
#热力图，查看Survived与其他特征间相关性大小
# plt.figure(figsize=(8, 8))
# sns.heatmap(fullSel[['Survived', 'Age', 'Embarked', 'Fare', 'Parch', 'Pclass',
#                     'Sex', 'SibSp', 'Title', 'FamilyNum', 'FamilySize', 'Deck',
#                      'TickCot', 'TickGroup']].corr(), cmap='BrBG', annot=True,
#            linewidths=.5)
# plt.xticks(rotation=45)
# plt.show()
fullSel = fullSel.drop(['FamilyNum', 'SibSp', 'TickCot', 'Parch'], axis=1)
#one-hot编码
fullSel = pd.get_dummies(fullSel)
PclassDf = pd.get_dummies(dataset['Pclass'], prefix='Pclass')
TickGroupDf = pd.get_dummies(dataset['TickGroup'], prefix='TickGroup')
familySizeDf = pd.get_dummies(dataset['FamilySize'], prefix='FamilySize')

fullSel = pd.concat([fullSel, PclassDf, TickGroupDf, familySizeDf], axis=1)
# # Create categorical values for Pclass
# dataset["Pclass"] = dataset["Pclass"].astype("category")
# dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
# # Drop useless variables
# dataset.drop(labels=["PassengerId", "Cabin", "Parch", "SibSp", "Name"], axis=1, inplace=True)
# # Separate train dataset and test dataset
# train = dataset[:train_len].copy()
# test = dataset[train_len:].copy()
# test.drop(columns="Survived", inplace=True)
# # Separate train features and label
# train.loc[:, ["Survived"]] = train["Survived"].astype(int)
# Y_train = train["Survived"].copy()
# X_train = train.drop(labels=["Survived"], axis=1)
experData = fullSel[fullSel['Survived'].notnull()]
preData = fullSel[fullSel['Survived'].isnull()]

X_train = experData.drop('Survived', axis=1)
Y_train = experData['Survived']
test = preData.drop('Survived', axis=1)
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms
random_state = 2
classifiers = []
########################################################################################################################
# classifiers.append(XGBClassifier(use_label_encoder=False, random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1))
# classifiers.append(RandomForestClassifier(oob_score=True, random_state=random_state))
# classifiers.append(ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=random_state))
# classifiers.append(GradientBoostingClassifier(random_state=random_state))
# classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression(random_state=random_state))
# classifiers.append(MultinomialNB())
# classifiers.append(SGDClassifier(random_state=random_state))
# classifiers.append(SVC(probability=True, random_state=random_state))
# ####################################################################################################
# classifiers.append(XGBClassifier(use_label_encoder=False, random_state=random_state, learning_rate=0.06, max_depth=4, min_child_weight=1.1, n_estimators=80, reg_alpha=0.015, reg_lambda=0.1))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state, max_depth=6, min_samples_split=10), random_state=random_state, learning_rate=0.009778, n_estimators=40))
# classifiers.append(ExtraTreesClassifier(oob_score=True, bootstrap=True, random_state=random_state, max_depth=8, max_features=9, min_samples_split=3, n_estimators=30))
# classifiers.append(GradientBoostingClassifier(random_state=random_state, learning_rate=0.09556, max_depth=3, min_samples_leaf=5, min_samples_split=0.015, n_estimators=160))
# classifiers.append(KNeighborsClassifier(n_neighbors=10, weights='uniform'))
# classifiers.append(LogisticRegression(random_state=random_state, C=0.7334, class_weight=None, solver='liblinear'))
# classifiers.append(MultinomialNB(alpha=4.7368))
# classifiers.append(SVC(probability=True, random_state=random_state, C=0.6, gamma=0.06, kernel='rbf'))
# ###############################################################################################################
#
# cv_results = []
# for classifier in classifiers:
#     cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring='accuracy', cv=kfold, n_jobs=-1))
#
# cv_means = []
# cv_std = []
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())
# #####################################################################################
# cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["XGBClassifier",
#         "AdaBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "KNeighboors",
#         "LogisticRegression", "MultinomialNB", "SGDClassifier", "SVC"]})
################################################################################################
# # cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["XGBClassifier",
# #         "AdaBoost", "ExtraTrees", "GradientBoosting", "KNeighboors",
# #         "LogisticRegression", "MultinomialNB", "SVC"]})
###################################################################################################
#g = sns.barplot(x="CrossValMeans", y="Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
#g.set_xlabel("accuracy")
#g = g.set_title("Cross validation scores")
# g = sns.FacetGrid(cv_res.sort_values(by='CrossValMeans', ascending=False), sharex=False,
#             sharey=False, aspect=2)
# g.map(sns.barplot, 'CrossValMeans', 'Algorithm', **{'xerr': cv_std},
#                palette='muted')
# g.set(xlim=(0.7, 0.9))
# g.add_legend()
# pd.set_option('display.max_rows', None)
# plt.show()
#pprint.pprint(cv_res.sort_values(by="CrossValMeans", ascending=False))
# preds = []
# for classifier in classifiers:
#     classifier.fit(X_train, Y_train)
#     preds.append(classifier.predict(test))
# # #############################################################################################################
# # # essembles = pd.DataFrame({"XGBC": preds[0], "AdaBoost": preds[1], "RandomForest": preds[2], "ET": preds[3],
# # #                           "GBC": preds[4], "KNN": preds[5], "LR": preds[6], "MLNB": preds[7], "SGDClassifier": preds[8],
# # #                           "SVC": preds[9]})
# # ###################################################################################################################
# # essembles = pd.DataFrame({"XGBC": preds[0], "AdaBoost": preds[1], "ET": preds[2],
# #                           "GBC": preds[3], "KNN": preds[4], "LR": preds[5], "MLNB": preds[6],
# #                           "SVC": preds[7]})
# #
# # # plt.figure(figsize=(6, 6))
# # g = sns.heatmap(essembles.corr(), annot=True, fmt=".2f")
# # g = plt.setp(g.get_xticklabels(), rotation=25)
# # plt.show()
# # # SVC
# # SVMC_best, SVMC_score, SVMC_params = search_best_SVC(X_train, Y_train, kfold, random_state)
# # print(SVMC_score, SVMC_params)
# # #GB
#GB_best, GB_score, GB_params = search_best_GBoost(X_train, Y_train, kfold, random_state)
#print(GB_score, GB_params)
# # #XGB
# # XGB_best, XGB_score, XGB_params = search_best_XGBoost(X_train, Y_train, kfold, random_state)
# # print(XGB_score, XGB_params)
#LR
LR_best, LR_score, LR_params = search_best_LogisticRegression(X_train, Y_train, kfold, random_state)
print(LR_score, LR_params)
# # #KNN
# # KNN_best, KNN_score, KNN_params = search_best_KNeighborsClassifier(X_train, Y_train, kfold)
# # print(KNN_score, KNN_params)
# # #adasboost
# # ada_best, ada_score, ada_params = search_best_AdaBoostClassifier(X_train, Y_train, kfold, random_state)
# # print(ada_score, ada_params)
# # #MLNB
# #MLNB_best, MLNB_score, MLNB_params = search_best_MultinomialNB(X_train, Y_train, kfold)
# #print(MLNB_score, MLNB_params)
# #MLNB
# #ExtC_best, ExtC_score, ExtC_params = search_best_ExtraTreesClassifier(X_train, Y_train, kfold, random_state)
# #print(ExtC_score, ExtC_params)
#
# # g1 = plot_learning_curve(classifiers[0], "XGBC learning curves", X_train, Y_train, cv=kfold)
# # g2 = plot_learning_curve(classifiers[1], "AdaBoost learning curves", X_train, Y_train, cv=kfold)
# # g3 = plot_learning_curve(classifiers[2], "ET learning curves", X_train, Y_train, cv=kfold)
# # g4 = plot_learning_curve(classifiers[3], "GBC learning curves", X_train, Y_train, cv=kfold)
# # g5 = plot_learning_curve(classifiers[4], "KNN learning curves", X_train, Y_train, cv=kfold)
# # g6 = plot_learning_curve(classifiers[5], "LR curves", X_train, Y_train, cv=kfold)
# # g7 = plot_learning_curve(classifiers[6], "MLNB learning curves", X_train, Y_train, cv=kfold)
# # g8 = plot_learning_curve(classifiers[7], "SVC learning curves", X_train, Y_train, cv=kfold)
# # plt.show()
#
# # nrows = ncols = 2
# # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))
# #
# # names_classifiers = [("XGBC", classifiers[0]), ("AdaBoosting", classifiers[1]), ("ExtraTrees", classifiers[2]), ("GradientBoosting", classifiers[3])]
# #
# # nclassifier = 0
# # for row in range(nrows):
# #     for col in range(ncols):
# #         name = names_classifiers[nclassifier][0]
# #         classifier = names_classifiers[nclassifier][1]
# #         indices = np.argsort(classifier.feature_importances_)[::-1][:10]
# #         g = sns.barplot(y=X_train.columns[indices][:10], x=classifier.feature_importances_[indices][:10], orient='h', ax=axes[row][col])
# #         g.set_xlabel("Relative importance", fontsize=12)
# #         g.set_ylabel("Features", fontsize=12)
# #         g.tick_params(labelsize=9)
# #         g.set_title(name + " feature importance")
# #         nclassifier += 1
# #
# # plt.show()
#
# votingC = VotingClassifier(estimators=[("XGBC", classifiers[0]), ("AdaBoosting", classifiers[1]), ("ExtraTrees", classifiers[2]),
#                                        ("GradientBoosting", classifiers[3]), ("KNN", classifiers[4]), ('LR', classifiers[5]),
#                                        ('MLNB', classifiers[6]), ('SVC', classifiers[7])], voting='hard', n_jobs=-1)
#
# print(cross_val_score(votingC, X_train, y=Y_train, scoring="f1", cv=kfold, n_jobs=-1).mean())
# #g = plot_learning_curve(votingC, "SVC learning curves", X_train, Y_train, cv=kfold)
# # #plt.show()
# # votingC = votingC.fit(X_train, Y_train)


#查看模型ROC曲线
#求出测试数据模型的预测值
modelgsGBCtestpre_y=LR_best.predict(X_train).astype(int)
#画图
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# Compute ROC curve and ROC area for each class
fpr, tpr, threshold = roc_curve(Y_train, modelgsGBCtestpre_y) ###计算真正率和假正率
roc_auc = auc(fpr, tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic GradientBoostingClassifier Model')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import confusion_matrix
print('GradientBoostingClassifier模型混淆矩阵为\n', confusion_matrix(Y_train.astype(int).astype(str),modelgsGBCtestpre_y.astype(str)))
#TitanicGBSmodle
GBCpreData_y = LR_best.predict(test)
GBCpreData_y = GBCpreData_y.astype(int)
#导出预测结果
GBCpreResultDf = pd.DataFrame()
GBCpreResultDf['PassengerId'] = dataset['PassengerId'][dataset['Survived'].isnull()]
GBCpreResultDf['Survived'] = GBCpreData_y
GBCpreResultDf
#将预测结果导出为csv文件
GBCpreResultDf.to_csv('TitanicGBSmodle1.csv', index=False)