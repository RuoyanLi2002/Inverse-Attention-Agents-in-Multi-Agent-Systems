import pickle
import random
import math
import numpy as np
a=np.zeros((1000,27))
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

    a[i][9] = float(random.uniform(-1+r, 1-r))
    a[i][10] = float(random.uniform(-1+r, 1-r))
    a[i][11] = 3.0

    a[i][12] = float(random.uniform(-1+r, 1-r))
    a[i][13] = float(random.uniform(-1+r, 1-r))
    a[i][14] = 4.0

    a[i][15] = float(random.uniform(-1+r, 1-r))
    a[i][16] = float(random.uniform(-1+r, 1-r))
    a[i][17] = 5.0

    a[i][18] = float(random.uniform(-1+r, 1-r))
    a[i][19] = float(random.uniform(-1+r, 1-r))
    a[i][20] = 6.0

    a[i][21] = float(random.uniform(-1+r, 1-r))
    a[i][22] = float(random.uniform(-1+r, 1-r))
    a[i][23] = 7.0

    a[i][24] = float(random.uniform(-1+r, 1-r))
    a[i][25] = float(random.uniform(-1+r, 1-r))
    a[i][26] = 8.0
    r1 = random.uniform(0, 0.025)
    theta = 2*math.pi*random.random()
    x1 = float(a[i][0] +r1*math.cos(theta))
    y1 = float(a[i][1]+ r1*math.sin(theta))
    
    L=[1,2,3,4,5,6,7,8]
    random.shuffle(L)
    a[i][L[0]*3]=x1
    a[i][L[0]*3+1]=y1

print(a[:5][:])
with open('chase_1_8_8class.pkl', 'wb') as f:
    pickle.dump(a, f)
