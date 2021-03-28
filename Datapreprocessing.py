import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

def preprocessing():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    combine_data = [train_data, test_data]
    combine_dropdata = []
    for data in combine_data:
        combine_dropdata.append(data.drop(["Cabin", "PassengerId"], axis=1))

    #combine_dropdata[0].describe(include=['O'])
    #combine_dropdata[0][["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
    #combine_dropdata[0][["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
    for dataset in combine_dropdata:
        dataset["Title"] = dataset.Name.str.extract("([A-Za-z]+)\.", expand=False)

    for dataset in combine_dropdata:
        dataset["Title"] = dataset["Title"].replace(
            ["Capt", "Col", "Countess", "Don", "Dr", "Jonkheer", "Lady", "Major", "Sir", "Dona"], "Aristocrat")
        dataset["Title"] = dataset["Title"].replace(["Mlle", "Ms"], "Miss")
        dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

# pd.crosstab(combine_dropdata[0]["Title"], combine_dropdata[0]["Survived"])
    #combine_dropdata[0][["Title", "Survived"]].groupby("Title", as_index=False).mean().sort_values(by="Survived", ascending=False)
    for data in combine_dropdata:
        data["Age"] = data.groupby("Title", as_index=False)["Age"].transform(lambda x: x.fillna(x.mean()))
        data["Group"] = data.duplicated(subset=["Ticket"], keep=False)
        data.Group = data.Group.replace({True:1, False:0})

#combine_dropdata[0].head()
    #combine_dropdata[0].info()
    #print("_"*40)
    #combine_dropdata[1].info()

#g = sns.FacetGrid(combine_dropdata[0], col="Survived")
#g.map(plt.hist, "Age", bins=20)
    #Data4 = combine_dropdata[0][["Age", "Survived"]].groupby("Age", as_index=False).mean()
    #Data4.plot("Age", "Survived", kind="scatter")

    for data in combine_dropdata:
        data.loc[data["Age"] < 15, "Age"] = 0
        data.loc[(data["Age"] < 55) & (data["Age"] >= 15), "Age"] = 1
        data.loc[data["Age"] >= 55, "Age"] = 2

    combine_dropdata[0] = combine_dropdata[0].drop(["Name", "Ticket"], axis=1)
    combine_dropdata[1] = combine_dropdata[1].drop(["Name", "Ticket"], axis=1)
    #combine_dropdata[1].head()

    #Data5 = combine_dropdata[0][["Age", "Survived"]].groupby("Age", as_index=False).mean()
    #Data5.plot("Age", "Survived", kind="scatter")

    #d = sns.FacetGrid(combine_dropdata[0], col="Embarked", height=2.2, aspect=1.6)
    #d.map(sns.barplot, "Survived", "Fare", alpha=.5, ci=None)
    #d.add_legend()
    #combine_dropdata[0][["Embarked", "Fare"]].groupby("Embarked", as_index=False).mean()
    #combine_dropdata[0][["Embarked", "Fare", "Pclass"]].groupby(["Embarked", "Pclass"], as_index=False).mean()
    combine_dropdata[1]["Fare"] = combine_dropdata[1].groupby(["Embarked", "Pclass"], as_index=False)["Fare"].transform(
        lambda x: x.fillna(x.mean()))

    #combine_dropdata[1].info()

    for data in combine_dropdata:
        data.loc[data["Fare"] < 14, "Fare"] = 0
        data.loc[data["Fare"] >= 14, "Fare"] = 1
    #combine_dropdata[0].head()

    for data in combine_dropdata:
        data["Familysize"] = data["SibSp"] + data["Parch"]

    #Data6 = combine_dropdata[0][["Familysize", "Survived"]].groupby(["Familysize"], as_index=False).mean()
    #Data6.plot("Familysize", "Survived", kind="scatter")

    for data in combine_dropdata:
        data.loc[data["Familysize"] == 0, "Familysize"] = 0
        data.loc[(data["Familysize"] > 0) & (data["Familysize"] < 4), "Familysize"] = 1
        data.loc[data["Familysize"] >= 4, "Familysize"] = 2

    combine_dropdata[0] = combine_dropdata[0].drop(["SibSp", "Parch", "Survived"], axis=1)
    combine_dropdata[1] = combine_dropdata[1].drop(["SibSp", "Parch"], axis=1)

    #combine_dropdata[1].head()

    cat_cols = ["Sex", "Title", "Embarked"]
    num_cols = ["Pclass", "Age", "Familysize", "Fare", "Group"]
    train_x = combine_dropdata[0]
    train_y = train_data["Survived"]
    test_x = combine_dropdata[1]
# data precessing
# numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
# ('scaler', StandardScaler())
# ])

    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))
                                        ])

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('one_hot', OneHotEncoder(handle_unknown='ignore'))
                                          ])

    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, num_cols),
                                               ('cat', categorical_transformer, cat_cols)
                                               ])

    return train_x, train_y, test_x, preprocessor