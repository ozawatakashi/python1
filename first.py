# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

a = 2
b = 3
c = 2+3
print(c)
#%%
import os
print(os.getcwd()) #directryの確認

os.chdir('/Users/ozawa/Desktop/python')
#%%


# リスト
d = [1,3,4,2]
e = [3,4,1,2]
f = d+e #dとeが繋がる
g = [[1,2],[3,4]]
h = [d,e,g]
#%%


# numpy: 行列を扱うパッケージ
import numpy as np

arr = np.array(d)
arr2 = np.array(e)
arr3 = np.array(f)

p = arr + arr2
#%%
# pandas データフレームを扱うパッケージ
import pandas as pd
# エクセルからデータファイル読み込む
# df = pd.read_csv('F-F_Research_Data_Factors_weekly.CSV')
xls = pd.ExcelFile('F-F_Research_Data_Factors_weekly.xls')
df = xls.parse(xls.sheet_names[0],header=4,index_col=0)
dfs = df.reset_index(drop=True)
#%%
# 回帰モデル
from sklearn.linear_model import LinearRegression
#%%

lm = LinearRegression()

x_data = df["Mkt-RF"]
y_data = df["SMB"]

xarr = np.array(x_data)
yarr = np.array(y_data)
Xarr = xarr.reshape(374, 1)
X_data = x_data.reshape(374,1)
#%%

lm.fit(X_data,y_data)


#%%
print(lm.coef_)
print(lm.intercept_)
#%%　散布図
import pylab
pylab.scatter(xarr, yarr, marker='.', linewidths=0)
pylab.grid(True)
pylab.xlabel('X')
pylab.ylabel('Y')
#%%
#回帰直線用のデータ生成
predict_x = np.arange(-25, 20, 1)
predict_y = lm.predict(predict_x.reshape(45, 1))
#%%
#回帰直線を描く
import matplotlib.pyplot as plt
plt.plot(xarr , yarr , 'o')

pylab.plot(predict_x, predict_y, 'r', linewidth=2)


pylab.show()