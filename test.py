#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:29:24 2019

@author: erikbornsma
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

name = 'finalData'
with open(name +'.csv', 'r') as f:      #There are some technical details about this I don't remember
    data = list(csv.reader(f))          #Converting to a list
    print(data)
    print(data[2:])
    #Headers = list(data)[0]             #Just in case we want to see/use header information
    times = []                          #Setting up array for time data
    xpositions = []                     #And position (x positions that is)
    ypositions = []                     #Holds the y positoins
    velocities = []                     #And velocity
    for row in data[2:]:                #Steps through all data rows (so, not the headers)
        times.append(float(row[0]))         #Builds time list. float makes sure the datatype is correct, also removes commas (csv originally!)
        xpositions.append(float(row[1]))     #Builds pos list
        ypositions.append(float(row[2]))    #Builds the y positions list
    
    #print(times)
    #print(positions)
    #Making an array and calculating average velocities
    for i in range(1,len(times)):       #Runs based on the size of the time array, i is an index. Needs to start at 1... See if you can see why
        v = (ypositions[i]-ypositions[i-1])/(times[i]-times[i-1]) #Calculates average velocity between points -- close to instant v
        velocities.append(v)                                    #Adds to v list
    velocities.append(velocities[len(velocities)-1])            #Adds the last point again -- so list is same length as position and time
            
#plt.scatter(times,velocities,c = times)
x = times
y = velocities

trend = np.polyfit(x,y,1)       #Fitting x and y data with a first degree polynomial -- a line!


plt.scatter(x,y,c = times)

#NEW
plt.plot(
        [x[0],x[-1]],                               #x values to plot the line
        [trend[1]   ,   trend[0]*x[-1]  +   trend[1] ],         #y values to plot the line
        color='red',                          #colors
        linewidth=3)                                #linewidth?
#plt.axhline(y=0, color = 'black')


plt.title('Position vs. Time Graph')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.show()
print('The Slope is: '+ str(    np.trunc(trend[0]*100)/100    ))
        


