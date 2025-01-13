import pickle
import random
import math
import numpy as np
a=np.zeros((1000,27))

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
    a[i][3] = float(a[i][0] +r1*math.cos(theta1))
    a[i][4] = float(a[i][1]+ r1*math.sin(theta1))
    a[i][5]=0.0
    a[i][6] = float(a[i][0] +r2*math.cos(theta2))
    a[i][7] = float(a[i][1]+ r2*math.sin(theta2))
    a[i][8]=0.0


    a[i][9] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][10] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][11]=1.0
    a[i][12] = float(a[i][9] +r5*math.cos(theta5))
    a[i][13] = float(a[i][10]+ r5*math.sin(theta5))
    a[i][14]=1.0
    a[i][15] = float(a[i][9] +r6*math.cos(theta6))
    a[i][16] = float(a[i][10]+ r6*math.sin(theta6))
    a[i][17]=1.0


    a[i][18] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][19] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][20]=2.0
    a[i][21] = float(a[i][18] +r3*math.cos(theta3))
    a[i][22] = float(a[i][19]+ r3*math.sin(theta3))
    a[i][23]=2.0
    a[i][24] = float(a[i][18] +r4*math.cos(theta4))
    a[i][25] = float(a[i][19]+ r4*math.sin(theta4))
    a[i][26]=2.0

print(a[:2][:])
with open('noshuffle_cluster3.pkl', 'wb') as f:
    pickle.dump(a, f)