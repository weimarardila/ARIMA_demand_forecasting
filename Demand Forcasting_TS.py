#!/usr/bin/env python
# coding: utf-8

# # Demand Forecasting Problem - Time Series Analysis

# In[1]:


import numpy as np
import pylab as pl
import pandas as pd
import time
from sklearn.model_selection import train_test_split


# ## Data Preparation

# In[497]:


#Importing the dataset
#Demand = pd.read_excel("C:\Users\weimar\Desktop\DB_2.xlsx", sheet_name='Clean Data')
Demand = pd.read_excel(r'D:\PhD\PhD Courses\8_Time Series\DB_2.xlsx', sheet_name='Time Series')
data = pd.DataFrame(Demand)


# In[498]:


##Dataset visualization
data


# ### Features

# In[500]:


#Product_List = pd.read_excel("C:\Users\weimar\Desktop\DB_2.xlsx", sheet_name='Product List')
Item_List = pd.read_excel(r'D:\PhD\PhD Courses\8_Time Series\DB_2.xlsx', sheet_name='Item List')
#Product list visualization sample
Item_List


# ## Summary Report

# ### Average

# In[501]:


#Training Set
Cumulative_demand = []
Size = []
Average_demand = []
for i in range (0, 316):
    Item = data.loc[data.Item_Code==Item_List.Item_Code[i],:]
    Demand_train, Demand_test = train_test_split(Item.Order_Demand, test_size=0.5, random_state=0)
    Cumulative_demand.append(sum(Demand_train))
    Size.append(len(Demand_train))
    
for i in range (0,316):
    Average_demand.append(Cumulative_demand[i]/Size[i])

#Test Set
Cumulative_demand_test = []
Size_test = []
Average_demand_test = []
for i in range (0, 316):
    Item = data.loc[data.Item_Code==Item_List.Item_Code[i],:]
    Demand_train, Demand_test = train_test_split(Item.Order_Demand, test_size=0.5, random_state=0)
    Cumulative_demand_test.append(sum(Demand_test))
    Size_test.append(len(Demand_test))
    
for i in range (0,316):
    Average_demand_test.append(Cumulative_demand[i]/Size[i])


# ### Variance

# In[502]:


#Trainig Set
Square_demand = []
Var_demand = []
for i in range (0,316):
    Item = data.loc[data.Item_Code==Item_List.Item_Code[i],:]
    Demand_train, Demand_test = train_test_split(Item.Order_Demand, test_size=0.5, random_state=0)
    Square_demand.append(sum(Demand_train*Demand_train))

for i in range (0,316):
    Var_demand.append((Square_demand[i]-Size[i]*Average_demand[i]*Average_demand[i])/(Size[i]-1))
    
#Test Set
Square_demand_test = []
Var_demand_test = []
for i in range (0,316):
    Item = data.loc[data.Item_Code==Item_List.Item_Code[i],:]
    Demand_train, Demand_test = train_test_split(Item.Order_Demand, test_size=0.5, random_state=0)
    Square_demand_test.append(sum(Demand_test*Demand_test))

for i in range (0,316):
    Var_demand_test.append((Square_demand_test[i]-Size_test[i]*Average_demand_test[i]*Average_demand_test[i])/(Size_test[i]-1))
len (Var_demand)


# ### Covariance

# In[503]:


Cov_demand = []
for i in range (0,316):
    Item = data.loc[data.Item_Code==Item_List.Item_Code[i],:]
    Demand_train, Demand_test = train_test_split(Item.Order_Demand, test_size=0.5, random_state=0)
    if len(Demand_train) == len(Demand_test):
        t = np.cov(Demand_train, Demand_test)
    else:
        t = np.cov(Demand_test[:(len(Demand_test)-1)],Demand_train)
    Cov_demand.append(np.take(t,1))
len(Cov_demand)


# ### Correlation

# In[504]:


Cor_demand =[]
n = 0
for i in range (0,316):
    Cor_demand.append(Cov_demand[i]/Var_demand[i])
    if abs(Cor_demand[i]) < 0.1:
        n=n+1
n


# ### Sumary Report

# In[505]:


Item_code = []
for i in range (0,316):
 Item_code.append(Item_List.Item_Code[i])


# In[506]:


Report = {'Item_Code': Item_code,'Mean': Average_demand, 'Variance': Var_demand, 'Covariance':Cov_demand, 'Correlation':Cor_demand  }
Report_data_frame = pd.DataFrame(data=Report)
Report_data_frame


# ## ARIMA Model

# In[507]:


from statsmodels.tsa.arima_model import ARIMA


# ### Model Stationarity

# #### Classification of the item list acording to the demand stationarity

# In[508]:


Stationary = []
Non_stationary = []
for i in range (0, 316):
 if abs(Report_data_frame.Correlation[i] )< 0.05 :
    Stationary.append(Report_data_frame.Item_Code[i])
 if abs(Report_data_frame.Correlation[i] )> 0.5 :
    Non_stationary.append(Report_data_frame.Item_Code[i])


# ### Parameters Estimation

# #### Non-Stationary

# In[509]:


#DATA PREPARTION 
m = []
m_test = []
for i in range (0, 15):
 Item = data.loc[data.Item_Code==Non_stationary[i],:]
 s = {'Date':Item.Month, 'Demand':Item.Order_Demand}
 Series_data_frame = pd.DataFrame(s)
 Series_data_frame.Date = pd.to_datetime(Series_data_frame.Date)
 Series_data_frame.set_index('Date', inplace=True)
 m.append(Series_data_frame)


# In[510]:


#Grpahical Representation
for i in range (0, 5):
 m[i].plot(figsize=(10,5), linewidth=5, fontsize=10)
 plt.xlabel('Year', fontsize=10);


# In[539]:


#Modeling
Nsta_prediction = []
order =(1,2,0)
for i in range (0,15):
 model = ARIMA(m[i],order)
 model_fit = model.fit()
 forecast = model_fit.predict(start = '2014-09-01', end='2015-06-01')
 Nsta_prediction.append(forecast)
 print (Non_stationary[i])
 print(model_fit.summary())


# #### Statationary

# In[512]:


w = []
for i in range (0, 74):
 Item = data.loc[data.Item_Code==Stationary[i],:]
 s = {'Date':Item.Month, 'Demand':Item.Order_Demand}
 Series_data_frame = pd.DataFrame(s)
 Series_data_frame.Date = pd.to_datetime(Series_data_frame.Date)
 Series_data_frame.set_index('Date', inplace=True)
 w.append(Series_data_frame)


# In[513]:


#Grpahical Representation
for i in range (0, 5):
 w[i].plot(figsize=(10,5), linewidth=5, fontsize=10)
 plt.xlabel('Year', fontsize=10);


# In[525]:


Sta_prediction = []
for i in range (0, 74):
 if i in range (29,40):
    order = (3, 0, 0)
 else:
    order = (3, 0, 1)
 Item = data.loc[data.Item_Code==Stationary[i],:]
 model = ARIMA(w[i], order)
 model_fit = model.fit()
 if i!=32:
  forecast = model_fit.predict(start = '2016-05-01', end='2017-02-01')
 else:
  forecast = model_fit.predict(start = '2014-09-01', end='2015-06-01')
 Sta_prediction.append(forecast)
 print (Stationary[i])
 print(model_fit.summary())


# ## Predictions

# In[531]:


Sta_prediction


# In[540]:


ST = pd.DataFrame(Non_stationary)


# In[547]:


ST2 = pd.ExcelWriter('List2.xlsx')
ST.to_excel(ST2,'Sheet1')
ST2.save()


# In[544]:


ST3 = pd.DataFrame(Nsta_prediction)
ST4 = pd.ExcelWriter('List1.xlsx')
ST3.to_excel(ST4,'Sheet2')
ST4.save()


# In[ ]:




