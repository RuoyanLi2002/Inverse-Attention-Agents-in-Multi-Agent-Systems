import pickle
import random
import math
import numpy as np
a=np.zeros((2000,12))
for i in range(2000):
    r = 0.1
    a[i][3] = float(random.uniform(-1+r, 1-r))
    a[i][4] = float(random.uniform(-1+r, 1-r))
    a[i][5] = 1.0
    a[i][6] = float(random.uniform(-1+r, 1-r))
    a[i][7] = float(random.uniform(-1+r, 1-r))
    a[i][8] = 2.0
    a[i][9] = float(random.uniform(-1+r, 1-r))
    a[i][10] = float(random.uniform(-1+r, 1-r))
    a[i][11] = 3.0
    while True:
        a[i][0] = float(random.uniform(-1+r, 1-r))
        a[i][1] = float(random.uniform(-1+r, 1-r))
        r1 = np.sqrt(np.sum(np.square(a[i][3:5] - a[i][:2])))
        r2 = np.sqrt(np.sum(np.square(a[i][6:8] - a[i][:2])))
        r3 = np.sqrt(np.sum(np.square(a[i][9:11] - a[i][:2])))
        if min([r1,r2,r3])>0.275:
            break
    

print(a[:5][:])
with open('boundary_learn_3.pkl', 'wb') as f:
    pickle.dump(a, f)

