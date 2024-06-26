import numpy as np
import apm_id as arx

######################################################
# Configuration
######################################################
# number of terms
ny = 2 # output coefficients
nu = 2 # input coefficients
# number of inputs
ni = 1
# number of outputs
no = 1
# load data and parse into columns
data = np.loadtxt('data.txt',delimiter=',')
######################################################

# generate time-series model
arx.apm_id(data,ni,nu,ny)
