import pickle
import random
import math
import numpy as np
a=np.zeros((1000,9))
for i in range(1000):
    r = 0.025
    a[i][0] = float(random.uniform(-1+r, 1-r))
    a[i][1] = float(random.uniform(-1+r, 1-r))
    a[i][2] = 0.0
    theta = 2*math.pi*random.random()
    r = random.uniform(0, 0.025)
    # if random.random()>0.5:
    #     theta = 0
    # else:
    #     theta = math.pi
    # theta = 0
    theta2 = theta+math.pi
    a[i][3] = float(a[i][0]+r*math.cos(theta))
    a[i][4] = float(a[i][1]+r*math.sin(theta))
    a[i][5] = 1.0
    a[i][6] = float(random.uniform(-1, 1))
    a[i][7] = float(random.uniform(-1, 1))
    a[i][8] = 2.0
print(a[:5][:])
with open('randomblue.pkl', 'wb') as f:
    pickle.dump(a, f)

