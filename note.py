import numpy as np
import matplotlib.pyplot as plt
from gekko.ML import Gekko_GPR,Gekko_SVR,Gekko_NN_SKlearn
from gekko.ML import Gekko_NN_TF,Gekko_LinearRegression
from gekko.ML import Bootstrap,Conformist,CustomMinMaxGekkoScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
u_control=np.zeros((2, 30))
result=np.arange(0,60).reshape(2,30)
print(result)
#u_control=np.append(result[:,1:],result[:,-1])

print(u_control)