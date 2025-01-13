# import numpy as np
# import pylab as pl
# import math

# x = np.array([ 2, 4, 8, 10, 12, 14, 16])
# y = np.array([ 5, 10, 15, 20, 25, 30, 35])
# angles = np.array([45,275,190,100,280,18,45]) 

# def draw_line(x,y,angle,length):
#   cartesianAngleRadians = (450-angle)*math.pi/180.0
#   terminus_x = x + length * math.cos(cartesianAngleRadians)
#   terminus_y = y + length * math.sin(cartesianAngleRadians)
#   pl.plot([x, terminus_x],[y,terminus_y])
# #   print [x, terminus_x],[y,terminus_y]


# pl.axis('equal')
# pl.axis([-5,40,-5,40])
# for i in range(0,len(x)):
# #   print x[i],y[i],angles[i]
#   draw_line(x[i],y[i],angles[i],10)

# pl.show()
# pl.savefig('books_read.png')

import matplotlib.pyplot as plt
import numpy as np
from mpe_runner import load_target_score11,load_target_score13,load_target_score16,load_target_score10
state_boundary = np.array([[0,0,0,0,0,1,0.2,0.3,2,0.3,-0.5,3]])
# boundary_score = load_target_score13(4, 10 ,is_state=True)
# state_boundary = np.array([[0,0,0,0,0,1,0,0,2,0,0,3]])
boundary_score = load_target_score11(4, 10 ,is_state=True)
adv_score = load_target_score16(3, 10 ,is_state=True)
state_adv = np.array([[0,0,0,-0.4,0.7,1,0.5,-0.1,2]])
state_food = np.array([[0,0,0,-0.3,-0.5,1,0.4,-0.5,2,0.1,0.9,3]])
food_score = load_target_score10(4, 10 ,is_state=True)


# result = boundary_score.get_score(state_boundary, 0.01,is_norm=False)
# Create a figure and axes
fig, ax = plt.subplots()
x = [0.7,0.1,0.2,0.3]
y=[0.9,0.3,0.4,0.5]
start = -0.95
end = 0.95



# Draw a vertical arrow
for i in range(20):
    for j in range(20):
        x_start = start+i*0.1
        y_start = start+j*0.1

        state_food[0,0:2]=[x_start,y_start]
        result = food_score.get_score(state_food, 0.01,is_norm=False)
        # state_adv[0,0:2]=[x_start,y_start]
        # result = adv_score.get_score(state_adv, 0.01,is_norm=False)
        # state_boundary[0,0:2]=[x_start,y_start]
        # result = boundary_score.get_score(state_boundary, 0.01,is_norm=False)
        scope = np.sqrt(np.sum(np.square(result[0])))
        # print(scope)
        result = result[0]/100
        print(result)
        x_end = x_start+result[0]
        y_end = y_start+result[1]
        print(x_end,y_end,x_start,y_start)
        ax.arrow(x_start, y_start, result[0], result[1], head_width=0.02, head_length=0.03, fc='black', ec='black')

# ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='black', ec='black')

# # Draw a horizontal arrow
# ax.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# # Draw a diagonal arrow
# ax.arrow(0, 0, 1, 1, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Show the plot
# pl.axis('equal')
plt.axis([-1,1,-1,1])
plt.show()
plt.savefig('books_read.png')