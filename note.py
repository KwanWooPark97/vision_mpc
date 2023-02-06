import numpy as np
import matplotlib.pyplot as plt
from gekko.ML import Gekko_GPR,Gekko_SVR,Gekko_NN_SKlearn
from gekko.ML import Gekko_NN_TF,Gekko_LinearRegression
from gekko.ML import Bootstrap,Conformist,CustomMinMaxGekkoScaler
import pandas as pd
from sklearn.model_selection import train_test_split
#Source function to generate data

def f(x):
    return np.cos(2*np.pi*x)

#represent noise from a data sample
N = 350
xl = np.linspace(0,1.2,N)
noise = np.random.normal(0,.3,N)
y_measured = f(xl) + noise
data = pd.DataFrame(np.array([xl,y_measured]).T,columns=['x','y'])
features = ['x']
label = ['y']
s = CustomMinMaxGekkoScaler(data,features,label)
ds = s.scaledData()
mma = s.minMaxValues()

print(mma)