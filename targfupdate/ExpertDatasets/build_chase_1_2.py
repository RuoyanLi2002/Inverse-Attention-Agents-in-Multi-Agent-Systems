import pickle
import random
import math
import numpy as np
a=np.zeros((1000,9))
for i in range(1000):
    r = 0
    a[i][0] = float(random.uniform(-1+r, 1-r))
    a[i][1] = float(random.uniform(-1+r, 1-r))
    a[i][2] = 0.0

    a[i][3] = float(random.uniform(-1+r, 1-r))
    a[i][4] = float(random.uniform(-1+r, 1-r))
    a[i][5] = 1.0

    a[i][6] = float(random.uniform(-1+r, 1-r))
    a[i][7] = float(random.uniform(-1+r, 1-r))
    a[i][8] = 2.0

    
    r1 = random.uniform(0, 0.025)
    theta = 2*math.pi*random.random()
    x1 = float(a[i][0] +r1*math.cos(theta))
    y1 = float(a[i][1]+ r1*math.sin(theta))
    
    L=[1,2]
    random.shuffle(L)
    a[i][L[0]*3]=x1
    a[i][L[0]*3+1]=y1

print(a[:5][:])
with open('chase_1_2.pkl', 'wb') as f:
    pickle.dump(a, f)

