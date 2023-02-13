import numpy as np
import matplotlib.pyplot as plt
from gekko.ML import Gekko_GPR,Gekko_SVR,Gekko_NN_SKlearn
from gekko.ML import Gekko_NN_TF,Gekko_LinearRegression
from gekko.ML import Bootstrap,Conformist,CustomMinMaxGekkoScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import deque
#Source function to generate data
x0=np.zeros((10,10,5))

x=deque(x0,maxlen=10)

print(x[0])