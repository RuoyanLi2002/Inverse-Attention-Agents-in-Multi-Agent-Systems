import pickle
import random
import math
import numpy as np
a=np.zeros((100000,36))
for i in range(5000):
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

    a[i][0] = float(x1 +r1*math.cos(theta1))
    a[i][1] = float(y1+ r1*math.sin(theta1))
    a[i][3] = float(x1 +r2*math.cos(theta2))
    a[i][4] = float(y1+ r2*math.sin(theta2))
    a[i][5]=0.0
    a[i][6] = float(x1 +r3*math.cos(theta3))
    a[i][7] = float(y1+ r3*math.sin(theta3))
    a[i][8]=1.0
    a[i][9] = float(x1 +r4*math.cos(theta4))
    a[i][10] = float(y1+ r4*math.sin(theta4))
    a[i][11]=1.0
    a[i][12] = float(x1 +r5*math.cos(theta5))
    a[i][13] = float(y1+ r5*math.sin(theta5))
    a[i][14]=2.0
    a[i][15] = float(x1 +r6*math.cos(theta6))
    a[i][16] = float(y1+ r6*math.sin(theta6))
    a[i][17]=2.0

    for j in range(18,36):
        a[i][j] = a[i][j-18]
    for j in range(6):
        a[i][3*(j+7)-1]=1.0

for i in range(5000,100000):
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
    a[i][6] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][7] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][8]=1.0
    a[i][9] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][10] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][11]=1.0
    a[i][12] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][13] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][14]=2.0
    a[i][15] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][16] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][17]=2.0

    # for j in range(18,36):
    #     a[i][j] = a[i][j-18]
    for j in range(6):
        r = np.sqrt(np.sum(np.square(a[i][0:2] -a[i][j*3:j*3+2])))

        if r<=0.7:
            a[i][(j+6)*3:(j+6)*3+2] = a[i][j*3:j*3+2]
            a[i][3*(j+7)-1]=1.0
        else:
            a[i][(j+6)*3:(j+6)*3+2] = [0,0]
            a[i][3*(j+7)-1]=0.0

# print(a[:5][:])
print(a[99990:100000][:])
with open('distill_alltogether_6.pkl', 'wb') as f:
    pickle.dump(a, f)