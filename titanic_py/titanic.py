#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 00:04:56 2016

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
os.chdir('/Users/ozawa/Desktop/python/titanic_py')
Original_train = pd.read_csv('train.csv')
Cabin = np.asarray( Original_train['Cabin'])
Cabin = pd.get_dummies(Cabin)
Cabin = Cabin.dot(np.ones(147))
# 使わない列削除
no_use_name = ['PassengerId','Name','Ticket','Cabin',]
Original_train = Original_train.drop(no_use_name,axis =1)           

# 教師データ生成
y_train = pd.get_dummies(Original_train['Survived'].astype(object))
y_train = np.asarray(y_train).astype(np.int32)

y_train = y_train[:,1]
Original_train = Original_train.drop('Survived',axis =1)

# ダミー変数生成
dtypes = Original_train.dtypes
names = list(dtypes[dtypes != 'float64'].index) # float型以外をダミー変数に
dum =pd.get_dummies(Original_train[names].astype(object)) # ダミー変数生成
x_train = Original_train
x_train = x_train.drop(names,axis =1) # ダミーにした変数を一旦除く       
x_train = pd.concat([x_train,dum,Cabin],axis = 1) # ダミー変数くっつける
# 前処理
x_train[np.isnan(x_train)] = 0 # 欠損値を0に変える。
x_train = np.asarray(x_train).astype(np.float32)

#%%
train = tuple_dataset.TupleDataset(x_train, y_train)
#%%

class TitanicChain(Chain):
    def __init__(self):
        super(TitanicChain, self).__init__(
            l1 = L.Linear(25,6),
            l2 = L.Linear(6,5), 
            l3 = L.Linear(5,2),
        )

    def __call__(self, x, train=True):
         h1 = F.sigmoid(self.l1(x))
         h2 = F.sigmoid(self.l2(h1))
         h3 = F.sigmoid(self.l3(h2))
         return h3
#%%
model1 = TitanicChain()
model = L.Classifier(model1)
optimizer = optimizers.Adam()
optimizer.setup(model)
#%%
train_iter = iterators.SerialIterator(train, 1)
test_iter = iterators.SerialIterator(train, 1,repeat=False, shuffle=False)
#%%
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (100, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(trigger = (1,'epoch')))
trainer.extend(extensions.PrintReport( ['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()

#%%
model1.l1.W.data
#%%
# titanicdataの読み込み
os.chdir('/Users/ozawa/Desktop/python/titanic_py')
Original_test = pd.read_csv('test.csv')
Cabin = np.asarray( Original_test['Cabin'])
Cabin = pd.get_dummies(Cabin)
Cabin = Cabin.dot(np.ones(76))



# 使わない列削除
no_use_name = ['PassengerId','Name','Ticket','Cabin']
PassengerId = Original_test.PassengerId
Original_test = Original_test.drop(no_use_name,axis =1)


# ダミー変数生成
dtypes = Original_test.dtypes
names = list(dtypes[dtypes != 'float64'].index) # float型以外をダミー変数に
dum =pd.get_dummies(Original_test[names].astype(object)) # ダミー変数生成
x_test = Original_test
x_test = x_test.drop(names,axis =1) # ダミーにした変数を一旦除く       
x_test = pd.concat([x_test,dum,Cabin],axis = 1) # ダミー変数くっつける
# 前処理
x_test = x_test.drop('Parch_9',axis =1)
x_test[np.isnan(x_test)] = 0 # 欠損値を0に変える。
x_test = np.asarray(x_test).astype(np.float32)
#%%
serializers.save_npz('model1', model)
serializers.save_npz('state1', optimizer)

#%%
xte  = Variable(x_test, volatile='on') 
yte = model.predictor(xte)
permmit = yte.data
permmit = permmit[:,1]
permmit[permmit > 0.5] = 1
permmit[permmit <= 0.5] = 0
permmit = pd.Series(permmit)
df = pd.DataFrame({ 'PassengerId':PassengerId ,
                        'Survived' : permmit})
df.to_csv("permmit.csv" )

