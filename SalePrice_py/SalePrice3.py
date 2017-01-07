#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:21:47 2016

@author: ozawa
"""

import os
import numpy as np
import pandas as pd
import time
#%%
# titanicdataの読み込み
os.chdir('/Users/ozawa/Desktop/python/SalePrice_py')
Original_train = pd.read_csv('train.csv')
Original_test = pd.read_csv('test.csv')

# 使わない列削除
no_use_name = ['Id','Fence','GarageCond','GarageFinish'
               ,'Functional','BsmtFinType2','BsmtFinType2'
               ,'BsmtExposure','ExterCond','Exterior2nd'
               ,'RoofStyle','HouseStyle','BldgType','Condition1'
               ,'LandSlope','LotConfig','Utilities','LandContour'
               ,'LotShape','Alley','Street']

Original_train = Original_train.drop(no_use_name,axis =1)                    
Id = Original_test.Id
Original_test = Original_test.drop(no_use_name,axis =1)           

#%%
#object型のデータを取り出す
dtypes1 = Original_train.dtypes
dtypes2 = Original_test.dtypes
names1 = list(dtypes1[dtypes1 == 'object'].index)
names2 = list(dtypes2[dtypes2 == 'object'].index)


#ダミー変数生成
dum1 =pd.get_dummies(Original_train[names1].astype(object)) # ダミー変数生成
dum2 =pd.get_dummies(Original_test[names2].astype(object)) # ダミー変数生成
dum3 = np.asarray(dum1).astype(np.int32) +1
#dum4 = dum3.reshape(1,1440,137)
#test,trainのダミー変数確認（片方にないものを消すため）
dumname1 = pd.DataFrame(np.asarray(dum1.columns).reshape(1,152))

#testデータにないダミー変数を抽出
dumname1.columns = list(np.asarray(dum1.columns))
dumname2 = list(np.asarray(dum2.columns))
nouse = dumname1.drop(dumname2,axis =1)
nouse = nouse.T
nouse = list(nouse[0])

# testデータにないダミー変数が有効になっている行の削除
Original_train = Original_train[dum1[nouse].apply(lambda x: sum(x), axis=1) == 0]


dum1 = dum1[dum1[nouse].apply(lambda x: sum(x), axis=1) == 0]
dum1 = dum1.drop(nouse,axis = 1)

usename = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF'
           ,'YearRemodAdd','YearBuilt'
           ,'Fireplaces','BsmtFinSF1','WoodDeckSF','LotArea','BsmtUnfSF','ScreenPorch']
x_train = Original_train[usename]
# x_train = x_train.drop(names1,axis =1) # ダミーにした変数を一旦除く       
x_train = pd.concat([x_train,dum1],axis = 1) # ダミー変数くっつける

x_train = x_train[~np.isnan(x_train).any(axis=1)] # 欠損値のある行を消す。

#%%
y_train = np.asarray(Original_train['SalePrice'].astype(np.int32))
y_train = np.asarray(Original_train['SalePrice'].astype(np.int32))
y_train = np.log(y_train)
Original_train = Original_train.drop('SalePrice',axis =1)


#%%
x_test = Original_test[usename]
# x_test = x_test.drop(names2,axis =1) # ダミーにした変数を一旦除く       
x_test = pd.concat([x_test,dum2],axis = 1) # ダミー変数くっつける

x_test[np.isnan(x_test)] = 0 # 欠損値を0に変える。
#%%
from scipy import *
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
 
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oti
import statsmodels.tsa.api as tsa
#%%
usename2 = usename
usename2.extend(dumname2)
usename = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF'
           ,'YearRemodAdd','YearBuilt'
           ,'Fireplaces','BsmtFinSF1','WoodDeckSF','LotArea','BsmtUnfSF','ScreenPorch']

varlst = ['SalePrice']
varlst.extend(usename2)
#%%
def fml_build(varlst):
    varlst.reverse()
    fml=varlst.pop()
    while len(varlst)>0:
        fml=fml+'+'+varlst.pop()
    return fml

#%%
eq =fml_build(usename2)
eq
eq2 = fml_build(usename)
#%%
y_train= pd.DataFrame({'SalePrice':y_train})
x_train.index =  [x for x in range(1440)]
y_train.index =  [x for x in range(1440)]
train= pd.concat([x_train,y_train],axis=1)

#%%
rlt=smf.ols('SalePrice ~'+
             '+OverallQual + GrLivArea+'+
             'GarageCars + TotalBsmtSF+'+
             'YearRemodAdd + YearBuilt +  BsmtFinSF1   + LotArea'+
             '+KitchenQual_Ex+KitchenQual_Fa+KitchenQual_Gd+KitchenQual_TA', data=train).fit()
#%%
rlt=smf.ols('SalePrice ~'+ eq2 +
            '+KitchenQual_Ex+KitchenQual_Fa+KitchenQual_Gd+KitchenQual_TA'+
            '+SaleCondition_Abnorml+SaleCondition_AdjLand+SaleCondition_Alloca+SaleCondition_Family+SaleCondition_Normal+SaleCondition_Partial'+
            '+SaleType_COD+SaleType_CWD+SaleType_Con+SaleType_ConLD+SaleType_ConLI+SaleType_ConLw+SaleType_New+SaleType_Oth+SaleType_WD'+
            '+PoolQC_Ex+PoolQC_Gd'+
            '+CentralAir_N+CentralAir_Y'+
            '+ExterQual_Ex+ExterQual_Fa+ExterQual_Gd+ExterQual_TA', data=train).fit()
#%%
rlt.summary()
#%%

submit = rlt.predict(x_test)
submit = np.exp(submit).reshape(1459)
submit = pd.Series(submit)
df = pd.DataFrame({ 'Id':Id ,
                        'SalePrice' : submit})
df.to_csv("submit4.csv" )

