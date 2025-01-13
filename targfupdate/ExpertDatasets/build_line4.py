import pickle
import random
import math
import numpy as np
a=np.zeros((1000,12))
for i in range(1000):
    L = [0,1,2,3]
    random.shuffle(L)
    r = 0.5
    x = float(random.uniform(-1+r, 1-r))
    y = float(random.uniform(-1+r, 1-r))
    j=0
    a[i][L[j]*3] = x
    a[i][L[j]*3+1] = y
    a[i][5] = 1.0
    a[i][8] = 2.0
    a[i][11] = 3.0
    r2 = float(random.uniform(0.25, 0.4))
    theta = 2*math.pi*random.random()
    theta3 = theta
    theta2 = theta+math.pi
    theta_all=[theta,theta2,theta3]
    r_all=[r2,r2,2*r2]
    for j in range(1,len(L)):
        a[i][L[j]*3] = float(x+r_all[j-1]*math.cos(theta_all[j-1]))
        a[i][L[j]*3+1] = float(y+r_all[j-1]*math.sin(theta_all[j-1]))

print(a[:5][:])
with open('lineup_4.pkl', 'wb') as f:
    pickle.dump(a, f)

