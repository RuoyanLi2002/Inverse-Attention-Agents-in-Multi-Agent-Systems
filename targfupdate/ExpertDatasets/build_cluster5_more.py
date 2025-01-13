import pickle
import random
import math
import numpy as np
a=np.zeros((1000,51))

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
    r9 = random.uniform(0, 0.025)
    r10 = random.uniform(0, 0.025)
    theta1 = 2*math.pi*random.random()
    theta2 = 2*math.pi*random.random()
    theta3 = 2*math.pi*random.random()
    theta4 = 2*math.pi*random.random()
    theta5 = 2*math.pi*random.random()
    theta6 = 2*math.pi*random.random()
    theta7 = 2*math.pi*random.random()
    theta8 = 2*math.pi*random.random()
    theta9 = 2*math.pi*random.random()
    theta10 = 2*math.pi*random.random()


    a[i][0] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][1] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][3] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][4] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][5]=0.0
    a[i][6] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][7] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][8]=0.0
    a[i][9] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][10] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][11]=0.0
    a[i][12] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][13] = float(random.uniform(-1+0.05, 1-0.05))
    a[i][14]=0.0
    L=[0,1,2,3,4]
    random.shuffle(L)
    # print(L)
    

    a[i][24] = float(a[i][L[0]*3] +r4*math.cos(theta4))
    a[i][25] = float(a[i][L[0]*3+1]+ r4*math.sin(theta4))
    a[i][26]=1.0
    a[i][27] = float(a[i][L[1]*3] +r5*math.cos(theta5))
    a[i][28] = float(a[i][L[1]*3+1]+ r5*math.sin(theta5))
    a[i][29]=1.0
    a[i][15] = float(a[i][L[2]*3] +r6*math.cos(theta6))
    a[i][16] = float(a[i][L[2]*3+1]+ r6*math.sin(theta6))
    a[i][17]=1.0
    a[i][18] = float(a[i][L[3]*3] +r7*math.cos(theta7))
    a[i][19] = float(a[i][L[3]*3+1]+ r7*math.sin(theta7))
    a[i][20]=1.0
    a[i][21] = float(a[i][L[4]*3] +r8*math.cos(theta8))
    a[i][22] = float(a[i][L[4]*3+1]+ r8*math.sin(theta8))
    a[i][23]=1.0
    random.shuffle(L)
    theta =[theta1,theta2,theta3,theta9,theta10]
    r = [r1,r2,r3,r9,r10]
    for j in range(5):
        a[i][30+j*3] = float(a[i][L[j]*3] +r[j]*math.cos(theta[j]))
        a[i][31+j*3] = float(a[i][L[j]*3+1]+ r[j]*math.sin(theta[j]))
        a[i][32+j*3]=2.0
    for j in range(2):
        a[i][45+j*3] = float(random.uniform(-1+0.05, 1-0.05))
        a[i][46+j*3] = float(random.uniform(-1+0.05, 1-0.05))
        a[i][47+j*3]=2.0

print(a[:2][:])
with open('cluster5_more.pkl', 'wb') as f:
    pickle.dump(a, f)