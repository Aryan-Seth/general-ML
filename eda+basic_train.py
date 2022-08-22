import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import normalize
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn


def do(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train.ravel())
    preds = model.predict(x_test)
    err = mean_squared_error(y_test, preds)**0.5
    sco = r2_score(preds,y_test)
    return err, sco
# returns mse on selected model


def call(a):
    if a == 1:
        model = LinearRegression()
        label = 'Linear Reg'
    if a == 2:
        model = SVR(C=170, gamma=5, epsilon=8)
        label = 'SVR'
    if a == 3:
        model = BayesianRidge()
        label = 'BayesianRidge'
    if a == 4:
        model = RandomForestRegressor()
        label = 'RFR'
    if a == 5:
        model = DecisionTreeRegressor(max_depth=10)
        label = 'DTR'
    if a == 6:
        model = GradientBoostingRegressor(n_iter_no_change=10)
        label = 'GBR'

    return label, model
# chooses model based on input and outputs label for plotting


def plotall(err, labels, n):
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x_pos, err, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('MSE per model')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('error for iterations= '+str(n))
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()
# plots mse of all selected models on given dataset


def dataclean(df):
    val = (df.iat[i, 0] for i in range(df.shape[0]))
    val = list(val)

    ord = []

    col = [i for i in range(0, len(df.columns))]
    df.columns = col
    col = list(col)

    intdata = df.select_dtypes("int")
    cont = df.select_dtypes("float")

    a = list(df.columns)

    for i in cont:
        b = pd.unique(df[i])
        for j in b:
            if(j >= 0):
                flag = 1
            else:
                flag = 0
                break
        if(flag == 0):
            print(str(i) + " is interval")
        else:
            print(str(i)+" is ratio")

    print(a)
    for l in a:
        p = len(pd.unique(df[l]))
        if p < 10:
            ord.append(l)

    end = int(len(a)-1)
    Y = df.iloc[:, end]
    df = df.drop([end], axis=1)
    print(Y)

    extracol = df.nunique(axis=1)
    sum_uniq = 0
    for i in range(len(extracol)):
        sum_uniq = sum_uniq+extracol[i]
    for i in ord:
        pd.concat([df, pd.get_dummies(df.iloc[:, i])], axis=1)
        df.drop([i])

    col = [i for i in range(0, len(df.columns))]
    df.columns = col
    print(df.head())
    imputer = KNNImputer(n_neighbors=3)

    X = imputer.fit_transform(df)
    df = pd.DataFrame(X)
    # df=normalize(df)
    df = pd.DataFrame(df)
    return Y, df
# converts ordinal data to one hot encoded, checks between ratio and interval data and imputes missing values


def fsps(x_train, y_train, x_test, model):

    sfs = SequentialFeatureSelector(model, n_features_to_select='auto')
    sfs.fit(x_train, y_train)
    print(sfs.get_support())
    x_train = sfs.transform(x_train)
    x_test = sfs.transform(x_test)
    return x_train, x_test
# implements feature selection for dataset after datacleaning


def histgrad(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, sample_weight=None)
    preds = model.predict(x_test)
    err = mean_squared_error(y_test, preds)**0.5
    sco = model.score(y_test, preds)
    return err, sco


def main():
    df = pd.read_csv("C:\\Users\\Aryan Seth\\Downloads\\supplementary_material_ilthermodataset_all.csv")
    # Y,df=dataclean(df)
    Y = df['Young']
    df.drop(['Young'], axis=1)
    imputer = KNNImputer(n_neighbors=3)
    # X = imputer.fit_transform(df)
    # df = pd.DataFrame(X)
    # print(df.head())
    # print(df.columns.values)
    df = normalize(df)
    df = pd.DataFrame(df)

    # print([df.iloc[:, i] for i in range(7, 10)])
    # print(Y.head())
    x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.3)
    y_train = y_train.values.reshape(-1, 1)
    # model = LinearRegression()
    # print(df.head())
    df = df.drop([0], axis=1)
    # x_train, x_test = fsps(x_train, y_train, x_test, model)

    k = int(input("how many models do you want to run?\n"))
    print("input 1 for ...")

    e = []
    s = []
    for j in range(0, 1):
        for i in range(0, k):
            model = None
            a = int(input())
            if a == 9:
                model = HistGradientBoostingRegressor
                histgrad(model, x_train, y_train, x_test, y_test)
            label1, model = call(a, x_train, y_train, x_test, y_test)
            er, sco = do(x_train, y_train, x_test, y_test, model)
            e.append(er)
            s.append(sco)
        
    print("test error= ")
    print(e)
    print("r2= ")
    print(s)
    # plotall(err, label, k)




def deep():
    df = pd.read_csv()
    Y, df = dataclean(df)
    print(df.head())
    x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.2)
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=19, out_features=500),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=500, out_features=200),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=200, out_features=100),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=100, out_features=20),                
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                        in_features=20, out_features=1),
        nn.ReLU()
    )
    model.double()
    new_shape = (613, 1)

    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    kl_weight = 0.1

    x_train = torch.tensor(x_train.values)
    y_train = torch.tensor(y_train.values)
    # x_train = x_train.view(new_shape)
    y_train = y_train.view(new_shape)
    for step in range(0, 20000):
        pre = model(x_train)
        mse = mse_loss(pre, y_train)
        kl = kl_loss(model)
        cost = mse + kl_weight*kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

# deep()

def plot():
    import seaborn as sns
    df = pd.read_csv("")
    Y, df = dataclean(df)
    df = df.iloc[:, 0:7]
    df = pd.concat([df, Y], axis=1)
    print(df.head())
    df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    print(df.head())
    g = sns.PairGrid(df, palette=True)
    g.map(sns.scatterplot)
    g.add_legend()

    plt.show()


def svroptim():
    df = pd.read_csv("")
    # Y,df=dataclean(df)
    Y = df['Young']
    df.drop(['Young'], axis=1)
    imputer = KNNImputer(n_neighbors=3)
    X = imputer.fit_transform(df)
    df = pd.DataFrame(X)
    df = normalize(df)
    df = pd.DataFrame(df)

    svr = SVR()
    param_grid = {
        'C': [1.1, 5.4, 170, 1001],
        'epsilon': [0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
        'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]
    }
    grid_search = GridSearchCV(estimator=svr, cv=5,    param_grid={
        'C': [1.1, 5.4, 170, 1001],
        'epsilon': [0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
        'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]
    }, n_jobs=-1, verbose=2)
    grid_search.fit(df, Y)
    print(grid_search.best_params_)
# svroptim()


def featimp():
    df = pd.read_csv("")
    # Y,df=dataclean(df)
    Y = df['Young']
    df=df.drop(['Young'], axis=1)
    imputer = KNNImputer(n_neighbors=3)
    X = imputer.fit_transform(df)
    df = pd.DataFrame(X)
    df = normalize(df)
    df = pd.DataFrame(df)
    print(df.columns.values)
    model = LinearRegression()
# fit the model
    model.fit(df, Y)
# get importance
    importance = model.coef_
# summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    
    
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.xticks([x for x in range(len(importance))],df.columns.values)
    pyplot.show()
