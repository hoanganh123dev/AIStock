#!/usr/bin/env python
# coding: utf-8

# In[1]:


# thu thập dữ liệu chứng khoán của AAPL
# tiền xử lý dữ liệu và huấn luyện kiểm tra
# tạo mô hình LSTM xếp chổng
# dự đoán dữ liệu thử nghiệm và vẽ biểu đồ đầu ra
# dự doán 30 ngày trong tương lai và vẽ biểu đồ đầu ra
import pandas_datareader as pdr


# In[2]:


# lấy api của web chứng khoán TIINGO 
import os
key= os.environ["TIINGO_API_KEY"] = "2b86c473bf09a2b702c378203ff8710b34324a13"


# In[3]:


# lấy dự liệu chứng khoán của APPLE từ web TIINGO
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')


# In[4]:


import pandas as pd


# In[5]:


# đọc dữ liệu chứng khoán của APPLE
df = pd.read_csv('AAPL.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df1=df.reset_index()['close']


# In[9]:


df1.shape


# In[10]:


df1


# In[11]:


# vẽ đồ thị mô tả dữ liệu là giá đóng(close)
import matplotlib.pyplot as plt
plt.plot(df1)


# In[12]:


import numpy as np


# In[13]:


#FE
# phạm vi nhất định trên tập huấn luyện là 0 và 1.
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
#  chia tỷ lệ tối đa tối thiểu
#phân cực
# chuyển đổi tỷ lệ phần trăm dao động của chỉ số kỹ thuật thành phạm vi [− 1, 1].
# biến đổi đầu vào thành mảng
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[14]:


df1


# In[15]:


# tách tập dữ liệu vào trong train và test split
#loại bỏ tính năng đệ quy bằng cách giảm quy mô dữ liệu đào tạo
# RFE
training_size=int(len(df1)*0.65)# kích thước của train có chiều dài của khung dữ liệu *0.65 = 65%
test_size=len(df1)-training_size# kích thước của test = chiều dài của khung dữ liệu  - train = 35%
#PCA
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[16]:


training_size,test_size


# In[17]:


test_data


# In[18]:


len(train_data), len(test_data)


# In[19]:


train_data


# In[24]:


import numpy
# chuyển đổi một mảng giá trị thành ma trận tập dữ liệu
#time_step là dấu thời gian ví dụ 100
def create_dataset(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0] #i=0, 0,1,2,3,4----- 100
        dataX.append(a)# phần tử thứ 3 sẽ là phần tử thứ 4
        dataY.append(dataset[i+time_step,0]) # bắt đầu từ phần tử 100 sau đó tới 101+++
    return numpy.array(dataX), numpy.array(dataY)


# In[25]:


# định hình lại thành X=t,t+1,t+2,t+3 và Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data,time_step)
X_test, y_test = create_dataset(test_data,time_step)


# In[26]:


# có 716 của 817 bản ghi để dự đoán các ngày sau
# bước này thuộc RFE
print(X_train.shape), print(y_train.shape)


# In[27]:


# có 340 của 441 bản ghi để dự đoán các ngày sau
print(X_test.shape), print(y_test.shape)


# In[28]:


#PCA
# định hình lại đầu vào thành [samples, time steps, features] rất cần thiết cho LSTM
# xử lý dữ liệu thành bộ tính năng đã hoàn thiện để đưa vào LSTM
# [0] = 71
#[1] = 100 chuyển đổi theo 2 hoặc 3 chiều
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)


# In[29]:


### tạo stacked cho mô hình LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[30]:


model=Sequential()# đầu vào cho mảng dữ liệu
model.add(LSTM(50, return_sequences=True,input_shape=(100,1)))# mô tả dữ liệu x_train và y_train tăng độ chính xác
model.add(LSTM(50,return_sequences=True))
#return_sequences=True khi xếp chồng các lớp LSTM để lớp LSTM thứ hai có đầu vào trình tự ba chiều
model.add(LSTM(50))
model.add(Dense(1))# input với đầu ra là 1 chiều
model.compile(loss='mean_squared_error',optimizer='adam')# tính sai số trung bình optimizer= tối ưu hóa tốc độ chạy của mô hình


# In[31]:


model.summary()


# In[32]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)
# lặp lại 100 lần 64 


# In[38]:


import tensorflow as tf


# In[39]:


### bắt đầu dự đoán và kiểm tra số liệu hiệu suất
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[40]:


### chuyển đổi trở lại dữ liệu gốc vì đã chuyển từ 0 1 giờ chuyển về lại để dự đoán
# phép biến đổi ngịch đảo vô hướng
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[41]:


### tính số liệu hiệu suất RMSE
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[42]:


### thử nghiệm dữ liệu RMSE
#R-MSE càng nhỏ tức là sai số càng bé thì mức độ ước lượng cho thấy độ tin cậy của mô hình có thể đạt cao nhất.
math.sqrt(mean_squared_error(y_test,test_predict))
# sai số 2 cái ko cao lắm


# In[43]:


train_predict


# In[44]:


test_predict


# In[37]:


### dự đoán train thay đổi để vẽ đồ thị
look_back=100 # bước thời gian
trainPredictPlot = numpy.empty_like(df1)#Trả về một mảng mới có cùng hình dạng và kiểu như một mảng đã cho.
trainPredictPlot[: , :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
### dự đoán test thay đổi để vẽ đồ thị 
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[: , :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
### vẽ đường cơ sở và dự đoán
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
# đầu ra là màu xanh lá 
# xanh lam là dữ liệu hoàn chỉnh
# màu cam là dữ liệu dùng để dự đoán 
# xanh lam sau xanh lá cam là đầu ra dự đoán


# In[45]:


len(test_data)


# In[46]:


#Nếu tôi xem xét ngày cuối cùng trong dữ liệu thử nghiệm là
#22-05-2020, tôi muốn dự đoán đầu ra của 23-05-2020. 
#Chúng tôi cần 100 dữ liệu trước đó để tôi lấy dữ liệu và định hình lại nó.
x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[47]:


# giờ lấy test_data để dự đoán 30 ngày sau
# lấy dữ liệu 100 ngày trc để dự đoán
temp_input=list(x_input)# chuyển nó thành 1 danh sách
temp_input=temp_input[0].tolist()# lấy tất cả các giá trị đó 


# In[48]:


temp_input


# In[49]:


# chứng minh dự đoán cho 30 ngày tới
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):# điều kiện là 30 ngày tới
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])# đầu vào bắt đầu từ 1 trở đi
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)# định hình lại dữ liệu
        x_input = x_input.reshape((1, n_steps, 1))# định hình lại dữ liệu
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)# bắt đầu dự đoán
        print("{} day output {}".format(i,yhat))# đầu ra 
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:# bất kể 100 kết quả nào nhận được chuyển tới mô hình 
        x_input = x_input.reshape((1, n_steps,1))# định hình lại dữ liệu
        yhat = model.predict(x_input, verbose=0)# thực hiện dữ đoán
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())# lấy tất cả các giá trị đó 
        print(len(temp_input))
        lst_output.extend(yhat.tolist())# thêm yhat vào lst_output 
        i=i+1
    

print(lst_output)


# In[50]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[51]:


import matplotlib.pyplot as plt


# In[52]:


len(df1)


# In[55]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))# chuyển về giá trị gốc phép biến đổi ngịch đảo df1


# In[56]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[57]:


df3=scaler.inverse_transform(df3).tolist()# chuyển đổi tỷ lệ


# In[58]:


plt.plot(df3)


# In[ ]:




