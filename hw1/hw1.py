
# coding: utf-8

# In[ ]:

### HW1_best
import time
import random
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv', encoding = "BIG5")
# train['日期'] = pd.to_datetime(train['日期'])
# date = pd.Series(train['日期'].drop_duplicates().loc[:])
# train['測項'] = train['測項'].astype('category')
x = train[train['測項'] == 'PM2.5'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
pm25 = x.values.flatten()

x = train[train['測項'] == 'SO2'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
SO2 = x.values.flatten()

x = train[train['測項'] == 'RH'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
RH = x.values.flatten()

x = train[train['測項'] == 'WS_HR'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
WS = x.values.flatten()

x = train[train['測項'] == 'AMB_TEMP'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
TEMP = x.values.flatten()

x = train[train['測項'] == 'PM2.5'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
pm25two = x.values.flatten() ** 2

x = train[train['測項'] == 'PM10'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
pm10 = x.values.flatten()

x = train[train['測項'] == 'WD_HR'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
WD = x.values.flatten()

x = train[train['測項'] == 'CO'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
CO = x.values.flatten()

x = train[train['測項'] == 'NO2'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
NO2 = x.values.flatten()

x = train[train['測項'] == 'O3'].drop(['日期','測站','測項'],1).astype(float)#.reset_index().drop(['index'],1)
O3 = x.values.flatten()

# treat with -1 in pm2.5
for i in range(len(pm25)):
    if(pm25[i] == -1):
        #print(i)
        for j in range(i-1,-1,-1):
            if pm25[j] != -1:
                pm25[i] = pm25[j]
                break
                
# treat with negative in CO
for i in range(len(CO)):
    if(CO[i] < 0):
        #print(i)
        for j in range(i-1,-1,-1):
            if CO[j] >= 0:
                CO[i] = CO[j]
                break

# treat with negative in SO2
for i in range(len(SO2)):
    if(SO2[i] < 0):
        SO2[i] = -SO2[i]

# treat negative temp in TEMP
for i in range(len(TEMP)):
    if(TEMP[i] <= 0 ):
        if (TEMP[i-12] <= 0):
            TEMP[i] = (TEMP[i-24] + TEMP[i+12])/2
        elif (TEMP[i+12] <= 0):
            TEMP[i] = (TEMP[i-12] + TEMP[i+24])/2
        else:
            TEMP[i] = (TEMP[i-12] + TEMP[i+12])/2

# 
originpm25 = pm25
# 
new = pm25 * pm10
new1 = CO*NO2*O3
new1 = new1 **2
pm25three = pm25 **3
# regularization
mean_pm25 = np.mean(pm25)
std_pm25 = np.std(pm25)
mean_SO2 = np.mean(SO2)
std_SO2 = np.std(SO2)
mean_RH = np.mean(RH)
std_RH = np.std(RH)
mean_WS = np.mean(WS)
std_WS = np.std(WS)
mean_TEMP = np.mean(TEMP)
std_TEMP = np.std(TEMP)
mean_pm25two = np.mean(pm25two)
std_pm25two = np.std(pm25two)
mean_new = np.mean(new)
std_new = np.std(new)
mean_WD = np.mean(WD)
std_WD = np.std(WD)
mean_CO = np.mean(CO)
std_CO = np.std(CO)
mean_NO2 = np.mean(NO2)
std_NO2 = np.std(NO2)
mean_O3 = np.mean(O3)
std_O3 = np.std(O3)
mean_new1 = np.mean(new1)
std_new1 = np.std(new1)
mean_pm25three = np.mean(pm25three)
std_pm25three = np.std(pm25three)
mean_pm10 = np.mean(pm10)
std_pm10 = np.std(pm10)

pm25 = (pm25 - mean_pm25) / std_pm25
SO2 = (SO2 - mean_SO2) / std_SO2
RH = (RH - mean_RH) / std_RH
WS = (WS - mean_WS) / std_WS
TEMP = (TEMP - mean_TEMP) / std_TEMP
pm25two = (pm25two - mean_pm25two) / std_pm25two
new = (new - mean_new) / std_new
WD = (WD - mean_WD) / std_WD
CO = (CO - mean_CO) / std_CO
NO2 = (NO2 - mean_NO2) / std_NO2
O3 = (O3 - mean_O3) / std_O3
new1 = (new1 - mean_new1) / std_new1
pm25three = (pm25three - mean_pm25three) / std_pm25three
pm10 = (pm10 - mean_pm10) / std_pm10

data = [pm25,SO2,RH,WS,TEMP,WD,pm10,new,CO,NO2,O3]

# training
x = []
y = []
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 5種污染物
        for t in range(len(data)):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(originpm25[480*i+j+9])
x = np.array(x)
y = np.array(y)

# add pm25 square term
x = np.concatenate((x, x ** 2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


l_rate = 10
iteration = 20000

#lamb = np.array([100,10,1,0,0.1,0.01,0.001,0.0001])
lamb = [0.0001]
w = np.zeros([len(lamb),len(x[0])])
cost_a = np.zeros(len(lamb))
cost = np.zeros(len(lamb))

# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
#wtest = np.zeros([len(lamb),len(x[0])])
#L = np.identity(len(x[0]))
#L[0][0] = 0
#for l in range(len(lamb)):
#    wtest[l] = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)+lamb[l]*L),x.transpose()),y)

for l in range(len(lamb)):
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))
    for i in range(iteration):
        hypo = np.dot(x,w[l])
        loss = hypo - y

        # Root Mean Square Error 
        cost[l] = np.sum(loss**2) / len(x)
        cost_a[l] = math.sqrt(cost[l])

        gra = 2*np.dot(x_t,loss)
        gra[1:len(gra)] = gra[1:len(gra)] + 2*lamb[l]*w[l][1:len(gra)]
        
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w[l] = w[l] - l_rate * gra/ada
        #print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save model
#for i in range(len(lamb)):
#    np.save('bestmodel%d.npy' %i,w[i])
    
# read model
# wlamb = np.load('bestmodel7.npy')
wlamb = w[0]


test = pd.read_csv(sys.argv[1], encoding = "BIG5",header = -1)
x = test[test[1] == 'PM2.5'].drop([0,1],1).astype(float)
pm25 = x.values.flatten()

x = test[test[1] == 'SO2'].drop([0,1],1).astype(float)
SO2 = x.values.flatten()

x = test[test[1] == 'RH'].drop([0,1],1).astype(float)
RH = x.values.flatten()

x = test[test[1] == 'WS_HR'].drop([0,1],1).astype(float)
WS = x.values.flatten()

x = test[test[1] == 'AMB_TEMP'].drop([0,1],1).astype(float)
TEMP = x.values.flatten()

x = test[test[1] == 'PM2.5'].drop([0,1],1).astype(float)
pm25two = x.values.flatten() ** 2

x = test[test[1] == 'PM10'].drop([0,1],1).astype(float)
pm10 = x.values.flatten()

x = test[test[1] == 'WD_HR'].drop([0,1],1).astype(float)
WD = x.values.flatten()

x = test[test[1] == 'CO'].drop([0,1],1).astype(float)
CO = x.values.flatten()

x = test[test[1] == 'NO2'].drop([0,1],1).astype(float)
NO2 = x.values.flatten()

x = test[test[1] == 'O3'].drop([0,1],1).astype(float)
O3 = x.values.flatten()


# treat with -1 in pm2.5
for i in range(len(pm25)):
    if(pm25[i] == -1):
        for j in range(i-1,-1,-1):
            if pm25[j] != -1:
                pm25[i] = pm25[j]
                break

# treat with negative in SO2
for i in range(len(SO2)):
    if(SO2[i] < 0):
        SO2[i] = -SO2[i]
        
# treat negative temp in TEMP
for i in range(len(TEMP)):
    if(TEMP[i] <= 0 ):
        if (TEMP[i-12] <= 0):
            TEMP[i] = (TEMP[i-24] + TEMP[i+12])/2
        elif (TEMP[i+12] <= 0):
            TEMP[i] = (TEMP[i-12] + TEMP[i+24])/2
        else:
            TEMP[i] = (TEMP[i-12] + TEMP[i+12])/2

new = pm25 * pm10   
new1 = CO*NO2*O3
new1 = new1 ** 2
pm25three = pm25 ** 3

pm25 = (pm25 - mean_pm25) / std_pm25
SO2 = (SO2 - mean_SO2) / std_SO2
RH = (RH - mean_RH) / std_RH
WS = (WS - mean_WS) / std_WS
TEMP = (TEMP - mean_TEMP) / std_TEMP
pm25two = (pm25two - mean_pm25two) / std_pm25two
new = (new - mean_new) / std_new
WD = (WD - mean_WD) / std_WD
CO = (CO - mean_CO) / std_CO
NO2 = (NO2 - mean_NO2) / std_NO2
O3 = (O3 - mean_O3) / std_O3
new1 = (new1 - mean_new1) / std_new1
pm25three = (pm25three - mean_pm25three) / std_pm25three
pm10 = (pm10 - mean_pm10) / std_pm10

data = [pm25,SO2,RH,WS,TEMP,WD,pm10,new,CO,NO2,O3]


test_x = []
for i in range(240):
    test_x.append([])
    # 5種污染物
    for t in range(len(data)):
        # 連續9小時
        for s in range(9):
            test_x[i].append(data[t][9*i+s])

test_x = np.array(test_x)
test_x = np.concatenate((test_x, test_x ** 2), axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    a = np.dot(wlamb,test_x[i])
    ans.append(a)

for i in range(len(ans)):
    if ans[i] < 0:
        ans[i] = 0
    
value = pd.DataFrame(np.asarray(ans).flatten())
id = test[0].drop_duplicates().reset_index().drop(['index'],1)
answer = pd.concat([id,value],axis = 1)
answer.columns = ['id','value']
answer.to_csv(sys.argv[2],index=None)

