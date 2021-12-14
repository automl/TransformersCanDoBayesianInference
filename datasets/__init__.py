import pandas as pd
import torch
import sklearn.datasets
import numpy as np
from catboost.datasets import titanic, amazon
import openml

def get_svmlight(name):
    data = sklearn.datasets.load_svmlight_file("datasets/"+name+".txt")
    X, y = data[0], (data[1] + 1) / 2
    sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
    pos = int(y.sum()) if y.mean() < 0.5 else int((1-y).sum())
    X, y = X[sort][-pos * 2:].todense(), y[sort][-pos * 2:]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    return X, y


def get_openml(did, max_samples):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    X = X[y < 2]
    y = y[y < 2]
    sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
    pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
    X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0])


def load_openml_list(dids, filter_for_nan=True, num_feats=100):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    # 40685 removed because of too few samples within classes 1&0
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        filtered = datalist[
        np.logical_and(datalist['NumberOfFeatures'] < num_feats, datalist['NumberOfInstancesWithMissingValues'] == 0)]
    else:
        filtered = datalist[datalist['NumberOfFeatures'] < num_feats]
    dids = filtered.did.values
    max_samples = 400

    for ds in filtered.index:
        entry = filtered.loc[ds]
        X, y, categorical_feats = get_openml(int(entry.did), max_samples)
        print(entry['name'], entry.did)

        datasets += [[entry['name'], X, y, categorical_feats]]

    return datasets, filtered


# Classification
valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]
test_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 41146, 41169, 41027, 23517, 41165, 41161, 41159, 40996, 41138, 1590, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]

def get_adult():
    iris_df = pd.read_csv('datasets/a1a.txt').drop(columns=['Id'])
    return X,y

def get_iris():
    iris_df = pd.read_csv('datasets/iris.csv').drop(columns=['Id'])
    y = iris_df['Species']
    y = y.map(lambda name: {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}[name])
    X = iris_df.drop(columns=['Species'])
    X = torch.tensor(X.to_numpy())
    y = torch.tensor(y.to_numpy())
    X = X.reshape(3,-1,X.shape[1]).transpose(0,1).reshape(-1,X.shape[1])
    y = y.reshape(3,-1).transpose(0,1).reshape(-1)
    return X,y


def get_2class_iris():
    iris_df = pd.read_csv('datasets/iris.csv').drop(columns=['Id'])
    y = iris_df['Species']
    y = y.map(lambda name: {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}[name])
    iris_df = iris_df[y < 2]
    y = y[y < 2]
    X = iris_df.drop(columns=['Species'])
    X = torch.tensor(X.to_numpy())
    y = torch.tensor(y.to_numpy())
    X = X.reshape(2,-1,X.shape[1]).transpose(0,1).reshape(-1,X.shape[1]).float()
    y = y.reshape(2,-1).transpose(0,1).reshape(-1)

    return X,y


def get_biochem(): # 6 features, 2 classes, 200 examples
    biochem_df = pd.read_csv('datasets/biochem.csv')
    y = biochem_df['class']
    y = y.map(lambda name: {'Abnormal': 0, 'Normal': 1,}[name])
    X = biochem_df.drop(columns=['class'])
    X = torch.tensor(X.to_numpy())
    y = torch.tensor(y.to_numpy())
    X = X.reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0])[:200].float()
    y = y.reshape(2, -1).transpose(0, 1).reshape(-1).flip([0])[:200]
    return X, y


def get_heart(): # 13 features, 2 classes, 274 examples
    biochem_df = pd.read_csv('datasets/heart.csv')
    y = biochem_df['target']
    X = biochem_df.drop(columns=['target'])
    X = torch.tensor(X.to_numpy())
    y = torch.tensor(y.to_numpy())
    X = X[:-1].reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0])[:274].float()
    y = y[:-1].reshape(2, -1).transpose(0, 1).reshape(-1).flip([0])[:274]
    return X, y

def get_2class_wine():
    # y is size 3 vector, actually integers but we pose it as regression problem
    X, y = sklearn.datasets.load_wine('datasets', return_X_y=True)
    X = X[y < 2][0:118]
    y = y[y < 2][0:118]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    return X, y

def get_wine():
    # y is size 3 vector, actually integers but we pose it as regression problem
    X, y = sklearn.datasets.load_wine('datasets', return_X_y=True)
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    return X, y

def get_breast_cancer():
    # y is binary
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    sort = np.argsort(y)
    X, y = X[sort][0:424], y[sort][0:424]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    return X, y

def get_cov_type(size=1000):
    # Actually has 7 classes
    X, y = sklearn.datasets.fetch_covtype(return_X_y=True)

    X = X[y < 3]#[0:118]
    y = y[y < 3] - 1#[0:118]
    sort = np.argsort(y)
    X, y = X[sort][0:423680], y[sort][0:423680]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    return X[0:size], y[0:size]

def get_titanic():
    # y is binary
    titanic_train, titanic_test = titanic()
    titanic_train = titanic_train.drop(columns=['Name']).drop(columns=['PassengerId'])
    titanic_train[titanic_train.select_dtypes(['object']).columns] = titanic_train[titanic_train.select_dtypes(['object']).columns].apply(lambda x: x.astype('category').cat.codes)
    titanic_train.fillna(-1,inplace=True)
    X = titanic_train.drop('Survived',axis=1).values
    y = titanic_train.Survived.values

    sort = np.argsort(y)
    X, y = X[sort][207:], y[sort][207:]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()

    return X, y

def get_amazon(size=1000):
    # y is binary
    data, _ = amazon()
    data[data.select_dtypes(['object']).columns] = data[data.select_dtypes(['object']).columns].apply(lambda x: x.astype('category').cat.codes)
    data.fillna(-1,inplace=True)
    X = data.drop('ACTION',axis=1).values
    y = data.ACTION.values

    sort = np.argsort(y)
    X, y = X[sort][0:3794], y[sort][0:3794]
    y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
    X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()

    return X[0:size], y[0:size]

# Regression


def get_housing(size=1000):
    # y in [.15,5.]
    X, y = sklearn.datasets.fetch_california_housing('datasets', return_X_y=True)
    return torch.from_numpy(X[0:size]), torch.from_numpy(y[0:size])

def get_boston():
    # y in [5.,50.]
    X, y = sklearn.datasets.load_boston('datasets', return_X_y=True)
    return torch.from_numpy(X), torch.from_numpy(y)

def get_diabetes():
    # y in [25.,346.], actually integers but we pose it as regression problem
    X, y = sklearn.datasets.load_diabetes('datasets', return_X_y=True)
    return torch.from_numpy(X), torch.from_numpy(y).float()

def get_linnerud():
    # y is size 3 vector, actually integers but we pose it as regression problem
    X, y = sklearn.datasets.load_linnerud('datasets', return_X_y=True)
    return torch.from_numpy(X), torch.from_numpy(y).float()


if __name__ == '__main__':
    X,y = get_biochem()
    print(f"biochem has X of shape {X.shape} and y of shape {y.shape}.")
