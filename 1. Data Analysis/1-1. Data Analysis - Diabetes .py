#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_diabetes # (1)
import numpy as np # (2,3)
from sklearn.model_selection import train_test_split # (4)


#     (1) 데이터 가져오기

# In[2]:


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
df_x = diabetes.data
df_y = diabetes.target

print(df_x.shape)
print(df_y.shape)


# In[3]:


print(diabetes.DESCR)


#     (2) 모델에 입력할 데이터 x 준비하기

# In[4]:


df_x = np.array(df_x)


#     (3) 모델에 입력할 데이터 y 준비하기

# In[5]:


df_y = np.array(df_y)


#     (4) train 데이터와 test 데이터로 분리하기
#    

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.2, random_state=42)


#     (5) 모델 준비하기

# In[7]:


W = np.random.rand(10)
b = np.random.rand()

print(W)
print(b)


# In[8]:


def model(X,W,b):
    predictions = 0
    for i in range(10):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions


#     (6) 손실함수 loss 정의하기

# In[9]:


def MSE(a, b):
    mse = ((a - b) ** 2).mean()
    return mse

def loss(x, w, b, y):
    predictions = model(x, w, b)
    L = MSE(predictions, y)
    return L


#     (7) 기울기를 구하는 gradient 함수 구현하기

# In[10]:


def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)
    
    # y_pred 준비
    y_pred = model(X, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db



# In[11]:


dW,db=gradient(df_x,W,b,df_y)
print(dW)
print(db)


#     (8) 하이퍼 파라미터인 학습률 설정하기

# In[12]:


LEARNING_RATE = 0.1


#     (9) 모델 학습하기

# In[13]:


losses = []

for i in range(1, 3001):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))


# In[14]:


import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


#     (10) test 데이터에 대한 성능 확인하기

# In[15]:


predictions = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
mse


#     (11) 정답 데이터와 예측한 데이터 시각화하기

# In[16]:


plt.scatter(X_test[:,0],y_test)
plt.scatter(X_test[:,0],predictions)
plt.show()


# In[ ]:




