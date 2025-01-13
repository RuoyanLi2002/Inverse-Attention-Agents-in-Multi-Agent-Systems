import pickle
import random
import math
import numpy as np
a=np.zeros((1000,24))
for i in range(1000):
    x1 = float(random.uniform(-1+0.05, 1-0.05))
    y1 = float(random.uniform(-1+0.05, 1-0.05))

    r1 = random.uniform(0, 0.025)
    r2 = random.uniform(0, 0.025)
    r3 = random.uniform(0, 0.025)
    r4 = random.uniform(0, 0.025)
    r5 = random.uniform(0, 0.025)
    r6 = random.uniform(0, 0.025)
    r7 = random.uniform(0, 0.025)
    r8 = random.uniform(0, 0.025)
    
    theta1 = 2*math.pi*random.random()
    theta2 = 2*math.pi*random.random()
    theta3 = 2*math.pi*random.random()
    theta4 = 2*math.pi*random.random()
    theta5 = 2*math.pi*random.random()
    theta6 = 2*math.pi*random.random()
    theta7 = 2*math.pi*random.random()
    theta8 = 2*math.pi*random.random()

    a[i][0] = float(x1 +r1*math.cos(theta1))
    a[i][1] = float(y1+ r1*math.sin(theta1))
    a[i][3] = float(x1 +r2*math.cos(theta2))
    a[i][4] = float(y1+ r2*math.sin(theta2))
    a[i][5]=0.0
    a[i][6] = float(x1 +r3*math.cos(theta3))
    a[i][7] = float(y1+ r3*math.sin(theta3))
    a[i][8]=0.0
    a[i][9] = float(x1 +r4*math.cos(theta4))
    a[i][10] = float(y1+ r4*math.sin(theta4))
    a[i][11]=0.0
    a[i][12] = float(x1 +r5*math.cos(theta5))
    a[i][13] = float(y1+ r5*math.sin(theta5))
    a[i][14]=1.0
    a[i][15] = float(x1 +r6*math.cos(theta6))
    a[i][16] = float(y1+ r6*math.sin(theta6))
    a[i][17]=1.0
    a[i][18] = float(x1 +r7*math.cos(theta7))
    a[i][19] = float(y1+ r7*math.sin(theta7))
    a[i][20]=1.0
    a[i][21] = float(x1 +r8*math.cos(theta8))
    a[i][22] = float(y1+ r8*math.sin(theta8))
    a[i][23]=1.0

print(a[:5][:])
with open('alltogether_8.pkl', 'wb') as f:
    pickle.dump(a, f)