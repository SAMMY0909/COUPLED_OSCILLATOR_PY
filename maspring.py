#!/usr/bin/python3
from pylab import figure, plot, xlabel, grid, legend, title, savefig
from matplotlib.font_manager import FontProperties
import numpy as np
import csv
from scipy.integrate import odeint
import matplotlib.animation as animation
import matplotlib.pyplot as plt
def statevec(s, t, c):
    x1, y1, x2, y2 = s
    m1, m2, k1, k2, L1, L2, b1, b2 = c
    s1= [y1,(-b1 * y1 - k1 *(x1 - L1) + k2*(x2 - x1 - L2)) / m1,y2,(-b2*y2-k2*(x2-x1 - L2)) / m2]
    return s1

# ODEINT to solve de of the sys
# Parameter val
# Masses
m1 = 1.0
m2 = 1.5
#Spring constants
k1 = 8.0
k2 = 40.0
# Natural lengths
L1 = 0.5
L2 = 1.0
# Friction coefficients
b1 = 0.8
b2 = 0.5
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities\n",
x1 = 0.5
y1 = 0.0
x2 = 2.25
y2 = 0.0
#Error tolerance limits
abserr = 1.0e-8
relerr = 1.0e-6
#Idefine stoptime
dur =10
#Idefine number of points
npoints = 250
#time_val colvec
t1 = [dur*float(i) / (npoints - 1) for i in range(npoints)]
#parameters_in_statevec
c = [m1, m2, k1, k2, L1, L2, b1, b2]
#supply initial pos array
pos0 = [x1, y1, x2, y2]
#apply odeint to get solutions by time
solnv = odeint(statevec, pos0, t1, args=(c,),atol=abserr, rtol=relerr)

#openfile to write data
with open('mpr.csv', 'w') as fp:
    for t1, w1 in zip(t1, solnv):
        fp.write(str(t1)), fp.write(","),fp.write(str(w1[0])), fp.write(","),fp.write(str(w1[1])),fp.write(","), fp.write(str(w1[2])),fp.write(","), fp.write(str(w1[3])), fp.write("\n")
#open_previous_file to read data
fl=open('mpr.csv','r')
#read in csv format
reader= csv.reader(fl)
rows=list(reader)
#get max_lines
num_lines=len(rows)
#initialise col vectors for state of the system
t1= np.zeros((num_lines-1,1))
x11=np.zeros((num_lines-1,1))
y11=np.zeros((num_lines-1,1))
x22=np.zeros((num_lines-1,1))
y22=np.zeros((num_lines-1,1))
j=0
#extract values from csv file and update positions and accelerations of the masses
for j in range(0,num_lines-1):
    s0=rows[j]
    #print(s0) for testing whether csv operation gives proper values
    #print(s0[0],s0[1],s0[2],s0[3],s0[4])
    t1[j][0]=s0[0]
    x11[j][0]=float(s0[1])
    y11[j][0]=float(s0[2])
    x22[j][0]=float(s0[3])
    y22[j][0]=float(s0[4])
#plot file to be saved as png using certain dimensions declared by figsize
figure(1, figsize=(6, 4.5))
#apply xlabel
xlabel('t')
#set gridlines
grid(True)
#set linewidth
lw = 2
#plot the displacement of two blocks on same graph
plot(t1, x11, 'b', linewidth=lw)
plot(t1, x22, 'g', linewidth=lw)
#put title and show legend
legend((r'$x_1$', r'$x_2$'), prop=FontProperties(size=16))
title('Mass Displacements x1 and x2 for the Coupled Spring-Mass System')
#save the image if you want to, of course you want to :P
savefig('xposmassspring.png', dpi=1200)
#plt.show()

#Ah, ate my head up as I dug up answers on stack echange to do this
fig, ax = plt.subplots()
#Combine two plots into one and display on same screen, note frames per sec should be same
ax.set_title("Pos vs time")
ax.set_xlabel("t")
ax.set_ylabel("Abs disp from init pos")
ax.grid(color='m', linestyle='-', linewidth=2)
line,=ax.plot(t1,x11,lw=3,color='r')
line1,=ax.plot(t1,x22,lw=3,color='b')
#defn to update the  column values to be passed to the plotter mp4/webAgg(done on chrome)
def update(num,x,y,z,line):
    line.set_data(x[:num], y[:num])#for first block position
    line.axes.axis([-1, 10,0,2.5])#change axes values to suit your own needs[x1,x2,y1,y2]
    line1.set_data(x[:num], z[:num])#for second block's position
    line.axes.axis([-1, 10,0,2.5])
    return line,line1
#build the animation, with as much swag stack exchange gives you before you run into problems with a dull face again, LOL
ani = animation.FuncAnimation(fig, update,len(t1), fargs=[t1, x11,x22,line],interval=25,blit=True)#
ani.save('test.mp4',writer='ffmpeg',extra_args=['-loglevel','verbose'])#You will run into problems if you don't have proper codecs installed,use debugging loglevel etc..
plt.show()#Ah, finally this shows up on webAgg server

