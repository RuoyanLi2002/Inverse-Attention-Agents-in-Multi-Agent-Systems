import pickle
import random
import math
import numpy as np
a=np.zeros((1000,9))
for i in range(1000):
    r = 0.5
    a[i][0] = float(random.uniform(-1+r, 1-r))
    a[i][1] = float(random.uniform(-1+r, 1-r))
    a[i][2] = 0.0
    r2 = float(random.uniform(0.25, 0.4))
    theta = 2*math.pi*random.random()
    # if random.random()>0.5:
    #     theta = 0
    # else:
    #     theta = math.pi
    # theta = 0
    theta2 = theta+math.pi
    a[i][3] = float(a[i][0]+r2*math.cos(theta))
    a[i][4] = float(a[i][1]+r2*math.sin(theta))
    a[i][5] = 1.0
    a[i][6] = float(a[i][0]+r2*math.cos(theta2))
    a[i][7] = float(a[i][1]+r2*math.sin(theta2))
    a[i][8] = 2.0
print(a[:5][:])
with open('lineup25_40.pkl', 'wb') as f:
    pickle.dump(a, f)

