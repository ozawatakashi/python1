#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 04:22:19 2016

@author: ozawa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:21:47 2016

@author: ozawa
"""

import os
import numpy as np
import pandas as pd
from chainer import  Function, gradient_check, Variable 
from chainer import optimizers,  utils
from chainer import Link, Chain, ChainList
from chainer import iterators, serializers
from chainer import report, training , datasets
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
import time
from chainer.training import extensions
#%%
# titanicdataの読み込み
os.chdir('/Users/ozawa/Desktop/python/SalePrice_py')
Original_train = pd.read_csv('train.csv')
Original_test = pd.read_csv('test.csv')

# 使わない列削除
no_use_name = ['Id']

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

#test,trainのダミー変数確認（片方にないものを消すため）
N = len(dum1.columns)
dumname1 = pd.DataFrame(np.asarray(dum1.columns).reshape(1,N))

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

x_train = Original_train.iloc[:,:79]
x_train = x_train.drop(names1,axis =1) # ダミーにした変数を一旦除く       
x_train = pd.concat([x_train,dum1],axis = 1) # ダミー変数くっつける
x_train[np.isnan(x_train)] = 0 # 欠損値を0に変える。

x_train = np.asarray(x_train).astype(np.float32)

#%%
y_train = np.asarray(Original_train['SalePrice'].astype(np.int32))
y_train = np.log(y_train)
Original_train = Original_train.drop('SalePrice',axis =1)

#%%
train = tuple_dataset.TupleDataset(x_train, y_train)


#%%
class PriceChain(Chain):
    def __init__(self):
        super(PriceChain, self).__init__(
            l1 = L.Linear(270,300),
            l2 = L.Linear(300,100),
            l3 = L.Linear(100,50),
            l4 = L.Linear(50,10),
            l5 = L.Linear(10,1)
            
        )

    def __call__(self, x):
         h1 = F.relu(self.l1(x))
         h2 = F.relu(self.l2(h1))
         h3 = F.relu(self.l3(h2))
         h4 = F.relu(self.l4(h3))
         h5 = self.l5(h4)
         return h5

#%%
class LossFuncL(Chain):
    def __init__(self, predictor):
        super(LossFuncL, self).__init__(predictor=predictor)

    def __call__(self ,x, t):
        t.data = t.data.reshape((-1,1)).astype(np.float32)

        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss':loss}, self)
        return loss

#%%
model0 = PriceChain()
model = LossFuncL(model0)
optimizer = optimizers.Adam()
optimizer.setup(model)
#%%
x_test = Original_test
x_test = x_test.drop(names2,axis =1) # ダミーにした変数を一旦除く       
x_test = pd.concat([x_test,dum2],axis = 1) # ダミー変数くっつける

x_test[np.isnan(x_test)] = 0 # 欠損値を0に変える。
x_test = np.asarray(x_test).astype(np.float32)
#%%
train_iter = iterators.SerialIterator(train, 100)
test_iter = iterators.SerialIterator(train, 100,repeat=False, shuffle=False)
#%%
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater,(1000, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(trigger = (10,'epoch')))
trainer.extend(extensions.PrintReport( ['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
#%%
xte  = Variable(x_test, volatile='on') 
yte = model.predictor(xte)
submit = yte.data
submit = np.exp(submit).reshape(1459)
submit = pd.Series(submit)
df = pd.DataFrame({ 'Id':Id ,
                        'SalePrice' : submit})
df.to_csv("submit3.csv" )

