import pickle
import random
import math
import numpy as np
a=np.zeros((2000,6))
for i in range(2000):
    r = 0.1
    a[i][3] = float(random.uniform(-1+r, 1-r))
    a[i][4] = float(random.uniform(-1+r, 1-r))
    
    while True:
        a[i][0] = float(random.uniform(-1+r, 1-r))
        a[i][1] = float(random.uniform(-1+r, 1-r))
        r2 = np.sqrt(np.sum(np.square(a[i][3:5] - a[i][:2])))
        if r2>0.2:
            break
    a[i][5] = 1.0

print(a[:5][:])
with open('boundary_learn.pkl', 'wb') as f:
    pickle.dump(a, f)

