import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sympy.stats import Variance

from metrics.metrics_MUPM import onhot_analysis
import scipy.stats as stats

def fit_1(x,y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    r_squared = model.score(x, y)
    print("Regression Coefficients (b1, b2, b3):", model.coef_)
    print("Intercept (b0):", model.intercept_)
    print(r_squared)
    return model,r_squared

def fit_2(x1,x2,y):
    X = np.vstack([x1,x2]).T
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    print("Regression Coefficients (b1, b2, b3):", model.coef_)
    print("Intercept (b0):", model.intercept_)
    print(r_squared)
    return model,r_squared

def fit_3(X,V):
    model = LinearRegression(fit_intercept=False)

    model.fit(X, V)
    r_squared = model.score(X, V)

    print("Regression Coefficients (b1, b2, b3):", model.coef_)
    print(r_squared)
    return model,r_squared

def convert_to_hashable(item):
    return tuple(arr.tobytes() if isinstance(arr, np.ndarray) else arr for arr in item)

def val(VI,VT,V,C,n_splits=1):
    combined = list(zip(VI,VT,V,C))
    random.shuffle(combined)
    split_size = len(VI) // n_splits
    trains = [combined[0:int(1.0*split_size)]]
    trains = trains
    tests = trains
    r2_train = []
    r2_test = []
    mae_train = []
    mae_test = []
    mse_train = []
    mse_test = []
    for train,test in zip(trains,tests):
        x1_train,x2_train,x3_train,y_train = [],[],[],[]
        x1_test, x2_test, x3_test, y_test = [], [], [], []
        for i in range(len(train)):
            x1_train.append(train[i][0])
            x2_train.append(train[i][1])
            x3_train.append(train[i][3])
            y_train.append(train[i][2])
        for i in range(len(test)):
            x1_test.append(train[i][0])
            x2_test.append(train[i][1])
            x3_test.append(train[i][3])
            y_test.append(train[i][2])

        #if x1_train.__len__() > 1:
        x1_train = np.concatenate(x1_train)
        x2_train = np.concatenate(x2_train)
        x3_train = np.concatenate(x3_train)
        y_train = np.concatenate(y_train)
        x1_test = np.concatenate(x1_test)
        x2_test = np.concatenate(x2_test)
        x3_test = np.concatenate(x3_test)
        y_test = np.concatenate(y_test)

        X_train = np.vstack([x1_train, x2_train, x3_train]).T
        X_test = np.vstack([x1_test, x2_test, x3_test]).T
        print(X_train.shape,y_train.shape)
        model, r_squared = fit_3(X_train, y_train)
        r2_train.append(r_squared)
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mae_train.append(mean_absolute_error(y_train, y_pred))
        mse_train.append(mean_squared_error(y_train, y_pred))
        mae_test.append(mean_absolute_error(y_test, y_pred_test))
        mse_test.append(mean_squared_error(y_test, y_pred_test))
        r2_test.append(r2_score(y_test, y_pred_test))

    print('r2_train:', r2_train)

    return r2_train,r2_test,mae_train,mae_test,mse_train,mse_test

def compute_uncer(VI,VT,V,C):
    combined = list(zip(VI, VT, V, C))
    random.shuffle(combined)
    com = combined[0:int(0.5*combined.__len__())]
    v1,v2,v3,cc = [],[],[],[]
    for item in com:
        v1.append(np.linalg.norm(item[0],ord=1))
        v2.append(np.linalg.norm(item[1],ord=1))
        v3.append(np.linalg.norm(item[2],ord=1))
        cc.append(np.linalg.norm(item[3],ord=1))
    print(np.mean(np.array(v1)),np.mean(np.array(v2)),
          np.mean(np.array(v3)),np.mean(np.array(cc)))
    print(np.std(np.array(v1)),np.std(np.array(v2)),
          np.std(np.array(v3)),np.std(np.array(cc)))
    return v1,v2,v3,cc

results_dir = ".../results.xlsx"
Variance_Img,Variance_Text,Variance_Img_Text,Covariance = (
    onhot_analysis(results_dir,20))

v1,v2,v3,cc = compute_uncer(Variance_Img,Variance_Text,Variance_Img_Text,Covariance)
r2_train,r2_test,mae_train,mae_test,mse_train,mse_test = val(Variance_Img,Variance_Text,Variance_Img_Text,Covariance)



