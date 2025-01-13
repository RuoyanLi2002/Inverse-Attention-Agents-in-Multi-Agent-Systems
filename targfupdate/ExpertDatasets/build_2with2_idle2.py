import pickle
import random
import math
import numpy as np
a=np.zeros((1000,18))

for i in range(1000):
    x1 = float(random.uniform(-1+0.05, 1-0.05))
    y1 = float(random.uniform(-1+0.05, 1-0.05))

    r1 = random.uniform(0, 0.025)
    r2 = random.uniform(0, 0.025)
    r3 = random.uniform(0, 0.025)
    r4 = random.uniform(0, 0.025)
    r5 = random.uniform(0, 0.025)
    r6 = random.uniform(0, 0.025)
    theta1 = 2*math.pi*random.random()
    theta2 = 2*math.pi*random.random()
    theta3 = 2*math.pi*random.random()
    theta4 = 2*math.pi*random.random()
    theta5 = 2*math.pi*random.random()
    theta6 = 2*math.pi*random.random()


    a[i][0] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][1] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][3] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][4] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][5]=0.0
    
    L=[0,1]
    random.shuffle(L)
    # print(L)

    a[i][9] = float(a[i][L[0]*3] +r4*math.cos(theta4))
    a[i][10] = float(a[i][L[0]*3+1]+ r4*math.sin(theta4))
    a[i][11]=1.0
    a[i][6] = float(a[i][L[1]*3] +r5*math.cos(theta5))
    a[i][7] = float(a[i][L[1]*3+1]+ r5*math.sin(theta5))
    a[i][8]=1.0

    a[i][12] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][13] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][14]=2.0
    a[i][15] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][16] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][17]=2.0

print(a[:2][:])
with open('2with2_idle2.pkl', 'wb') as f:
    pickle.dump(a, f)