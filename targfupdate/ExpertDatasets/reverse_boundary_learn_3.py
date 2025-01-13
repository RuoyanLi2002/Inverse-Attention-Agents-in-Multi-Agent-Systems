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
    L=[1,2,3,4,5]
    random.shuffle(L)
    r1 = random.uniform(0, 0.1)
    theta1 = 2*math.pi*random.random()
    choose = L[0]
    if choose<4:
        x1 = a[i][3*choose]
        y1 = a[i][3*choose+1]
        a[i][0] = float(x1 +r1*math.cos(theta1))
        a[i][1] = float(y1+ r1*math.sin(theta1))
    else:

        x1 = float(random.uniform(-1, -0.9))
        x2 = float(random.uniform(0.9, 1))
        y1 = float(random.uniform(-1, -0.9))
        y2 = float(random.uniform(0.9, 1))
        tx = [x1,x2]
        ty = [y1,y2]
        random.shuffle(tx)
        random.shuffle(ty)
        a[i][0] = tx[0]
        a[i][1] = ty[0]


print(a[:5][:])
with open('reverse_boundary_learn_3.pkl', 'wb') as f:
    pickle.dump(a, f)

