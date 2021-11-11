"""
To do:
-- TEST
"""

#Packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from csv import reader
# import os

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

datafile = "one.csv"

with open(datafile,'r') as f:
    data = list(reader(f,delimiter=','))
    
    #A column-based csv file sort
    expParams, tData, posxData, posyData, poszData = [],[],[],[],[]
    ccc = 0
    for row in data:
        if ccc == 0:
            for thing in row[0:3]:
                expParams.append(float(thing))
                ccc+=1
        else:
            tData.append(float(row[0]))
            posxData.append(-float(row[1]))
            posyData.append(0)
            poszData.append(float(row[2]))
            
#Graphing Settings
Axis = ['posx','posy','posz']
xMin,xMax = 0,1.0       #The autoscale can make the graph look strange. This is the override
yMin,yMax = -.5,.5  
zMax = 1              #zMin is set to be the floor below
camAngle1, camAngle2 = 45, 90  

#Allows for two parameter scanning. Must use correct reference.
#Script will plot all combinations (so 2 and 3 means 6 trajectories)
scans = {'wind':[0,1],
         'omgy':[0,-1.,1.]}
scanList = []
for var in scans:scanList.append(var)

#Launch Settings
v0 = expParams[1]         #in m/s, initial speed
theta = 0       #in degrees, angle of vi from x axis
phi = expParams[0]        #in degrees, initial angle from x-y plane 

#Wind Settings
vW = 5.          #Windspeed relative to ground, in m/s
thetaW = 330.      #Angle of wind relative to x degrees
phiW = 0.        #Angle of wind relative to x-y plane in degrees
    
#Model Parameters
gx = 0 
gy = 0
gz = -9.8      #units of m/s/s, directed downward
floor = -0.01
zMin = floor
m = .5          #mass of projectile in kg
r = .21
A = np.pi*r*r   #area of projectile in m^2
Cd = 0.4       #drag coefficient
Cl = 0.25       #Lift coeff for new magnus
rho = 1.4       #density of air in kg/m^3  
dynVisc = 1.7*10**(-5) #Absolute/dynamic viscosity of air (Pa s) http://www.engineeringtoolbox.com/dry-air-properties-d_973.html
vdConst = 500.0     #Constant of proportionality for viscous drag (temp dependence?)
Dt = tData[3]-tData[2]      #Units of seconds
maxTime = 20    #To stop the while loop in case of disaster
vVec = [
    v0*np.cos(phi*np.pi/180)*np.cos(theta*np.pi/180),
    v0*np.cos(phi*np.pi/180)*np.sin(theta*np.pi/180),
    v0*np.sin(phi*np.pi/180)
]
vWVec = [    #Calculates and holds absolute wind vector values
    vW*np.cos(phiW*np.pi/180)*np.cos(thetaW*np.pi/180),
    vW*np.cos(phiW*np.pi/180)*np.sin(thetaW*np.pi/180),
    vW*np.sin(phiW*np.pi/180)
]

#Radconvert
#OLD radSec = 740*2.*np.pi/60
#OLD omgVec = [0.01,0.01,1.0]  #Measured in rad/sec

omgMag = expParams[2]*10 #Magnitude of spin in rad/sec
omgTheta = 90   #Angle of angular velociy from x-axis in degrees
omgPhi = 0     #Angle of angular velocity from x-y-plane in degrees
omgVec = [
    omgMag*np.cos(omgPhi*np.pi/180)*np.cos(omgTheta*np.pi/180),
    omgMag*np.cos(omgPhi*np.pi/180)*np.sin(omgTheta*np.pi/180),
    omgMag*np.sin(omgPhi*np.pi/180)
    ]

#Allows for clean axis labeling
niceNames = {
    'posx': '$x$ position (m)',
    'posy': '$y$ position (m)',
    'posz': '$z$ position (m)',
    'vx':   '$x$ velocity (m/s)',
    'vy':   '$y$ velocity (m/s)',
    'vz':   '$z$ velocity (m/s)',
    'omgx': '$x$ ang vel (rad/sec)',
    'omgy': '$y$ ang vel (rad/sec)',
    'omgz': '$z$ ang vel (rad/sec)',
    'vwRx': '$x$ R windspeed (m/s)',
    'vwRy': '$y$ R windspeed (m/s)',
    'vwRz': '$z$ R windspeed (m/s)',
    'time': 'Time (s)',
    'accelx': '$x$ acceleration (m/s/s)',
    'accely': '$y$ acceleration (m/s/s)',
    'accelz': '$z$ acceleration (m/s/s)',
    'Dt':   'time step (s)',
    'wind': 'Existance of wind (bin)'    
    }

#Defining initial state
state = {
    'num': 0,   #A numbering of the state    
    'posx': 0,  #in m, initial horizontal x position
    'posy': 0,  #in m, initial horizontal y position    
    'posz': 0,  #in m, initial vertical position
    'vx':   vVec[0],
    'vy':   vVec[1],
    'vz':   vVec[2],
    'omgx': omgVec[0],
    'omgy': omgVec[1],
    'omgz': omgVec[2],
    'vwRx': vWVec[0]-vVec[0],
    'vwRy': vWVec[1]-vVec[1],
    'vwRz': vWVec[2]-vVec[2],    
    'time': 0,    #in seconds, current time
    'accelx':gx,
    'accely':gy,
    'accelz':gz,
    'Dt':   Dt,
    'wind': 1,
    'rNum': 100,
    'coDrag': Cd,
    'coLift': Cl
    }

#Make a list of keys in the correct order to feed parameters. Might be a better way to do this...
keyList = ['posx','posy','posz','vx','vy','vz','omgx','omgy',
           'omgz','vwRx','vwRy','vwRz','time','accelx','accely',
           'accelz','Dt','wind','rNum','coDrag','coLift','num']

#Creates the text box listing initial parameters
textBox1 = "Initial State:\n---------"
for x in state:
    textBox1 = textBox1 + '\n' + x + ': ' + str(np.round(state[x],2))

#Sets up the objects. One contains current state information, the other lists of data
launches = {}   #Dictionary which holds all the different starting states (which are themselves dictionaries)
launchData = {} #Dictionary of dictionaries which hold lists of data for each parameter 
counter = 0     #Nuts and bolts...
landInfo = [''] #Used later to annotate the landing positions
for param1 in scans[scanList[0]]:
    for param2 in scans[scanList[1]]:
        #This one is tricky. Makes the dictionary of dictionaries containing starting states for different launches
        counter +=1
        reference = (scanList[0]+' = '
            +str(param1)+',\n'+scanList[1]
            +' = '+str(param2))      #Name for each different launch (e.g. 'wind = 0' and 'wind = 1')
        launches[reference] = state.copy()  #Makes a cheap copy of the state so I can run a different analysis on it.
        launches[reference][scanList[0]] = param1     #Changes the scanning variable
        launches[reference][scanList[1]] = param2     #Changes the 2nd scanning variable
        launches[reference]['num'] = counter
        
        #Sets up dictionaries to record data for the different parameters    
        launchData[reference] = {}
        for key in keyList: 
            launchData[reference][key] = []
            
        #Sets up a list we can use to populate landing data
        landInfo.append('')

#Definitions of functions used inside of the state-stepping function   
def stepx(x, v):        #Euler approximation of new position based on current velocity
    newx = x + v*Dt
    return newx
    
def stepv(v,a):         #Euler approximation of new velocity based on current accleration
    newv = v + a*Dt
    return newv

def stepa(g, vwR1,omg1, wind,omg2,omg3,vwR2,vwR3,coDrag,coLift):    
    #stepa(gx,vwRx,omgx,wind,omgy,omgz,vwRy,vwRz,coDrag,coLift)
    #Uses N2L to find the new acceleration. Last term on right is for direction
    if wind == 1:           #Allows us to ignore the effects of air resistance selectively.
        if vwR1 == 0:
            vwR1 = 0.000001
        if vwR2 == 0:
            vwR2 = 0.000001
        V = np.sqrt(vwR1**2+vwR2**2+vwR3**2)                #For new magnus calculation
        W = np.sqrt(omg1**2+omg2**2+omg3**2)                #For new magnus calculation
        newa = (1.0/m)*(
            m*g +                                           #Gravity
            .5*rho*vwR1**2*coDrag*A*(vwR1/np.abs(vwR1)) -    #Linear drag
            0.5*rho*A*coLift*V/W*((omg2*vwR3)-(omg3*vwR2))     #Magnus new
            #-np.pi**2*r**3*rho*(omg2*vwR3-omg3*vwR2))        #Magnus old
            )
    else:
        newa = g
    return newa

def stepvw(v,vW):           #Calculates new relative windspeed
    newvw = vW-v
    return newvw

def findReynolds(vwRx,vwRy,vwRz):
    relV = np.sqrt(vwRx**2+vwRy**2+vwRz**2)    
    rNum = 2.*r*relV*rho/dynVisc
    return rNum
    
def newDrag(rNum):
    coDrag = (24/rNum +
        2.6*(rNum/5)/(1+(rNum/5)**1.52) +
        0.411*(rNum/(2.63*(10**5))**(-7.94))/(1+(rNum/(2.63*(10**5))**(-8))) +
        0.25*(rNum/10**6)/(1+(rNum/(10**6)))
        )
    return coDrag

def newLift(coLift,omg1,omg2,omg3):
    W = np.sqrt(omg1**2+omg2**2+omg3**2)    
    coLift = Cl*2 #(3.19*10**(-1))*(1-np.exp((-2.48*10**(-3))*W))
    return coLift
    
def stepOmg(omg,wind):
    if wind == 1:
        if omg == 0:
            omg = 0.1
        newOmg = omg - Dt*(vdConst*dynVisc*r**2*omg/m)/(np.sqrt(dynVisc*rho/np.abs(omg)))
    else:
        newOmg = omg
    return newOmg
       
def updateState(x,y,z,vx,vy,vz,omgx,omgy,omgz,vwRx,vwRy,vwRz,t,ax,ay,az,Dt,wind,rNum,coDrag,coLift,num):
    newState = {
        'posx':     stepx(x, vx),
        'posy':     stepx(y, vy),
        'posz':     stepx(z, vz),
        'vx':       stepv(vx, ax),
        'vy':       stepv(vy, ay),
        'vz':       stepv(vz, az),
        'omgx':     stepOmg(omgx,wind),
        'omgy':     stepOmg(omgy,wind),
        'omgz':     stepOmg(omgz,wind),
        'vwRx':     stepvw(vx,vWVec[0]),
        'vwRy':     stepvw(vy,vWVec[1]),
        'vwRz':     stepvw(vz,vWVec[2]),
        'time':     t + Dt,
        'accelx':   stepa(gx,vwRx,omgx,wind,omgy,omgz,vwRy,vwRz,coDrag,coLift),
        'accely':   stepa(gy,vwRy,omgy,wind,omgz,omgx,vwRz,vwRx,coDrag,coLift),
        'accelz':   stepa(gz,vwRz,omgz,wind,omgx,omgy,vwRx,vwRy,coDrag,coLift),
        'Dt':       Dt,
        'wind':     wind,
        'rNum':     findReynolds(vwRx,vwRy,vwRz),
        'coDrag':   newDrag(rNum),
        'coLift':   newLift(coLift,omgx,omgy,omgz),
        'num':      num
        }  
    return newState
   
#The loop which goes the real work!
for launch in launches:
    while (launches[launch]['posz'] > floor and launches[launch]['time'] < maxTime): 
        #Looks until we hit ground level -- or we time out!    
        paramList = []      #Clearing list to use to hold current state parameters
        
        for datatype in keyList:  #This whole loop just locks down information on to the data list
            launchData[launch][datatype].append(launches[launch][datatype])  #Write to data
            paramList.append(launches[launch][datatype])       #Write to short-term param list  
            
        launches[launch] = updateState(*paramList) #Does the work! * "unpacks" the array to feed function
    
#Now the plotting!
font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 14}
fig = plt.figure(figsize=(10,10))
mpl.rcParams['legend.fontsize'] = 8
ax = fig.add_subplot(111, projection='3d')

#Plots themselves
for launch in launches:
    ax.plot(launchData[launch][Axis[0]], 
    launchData[launch][Axis[1]],
    launchData[launch][Axis[2]], 
    label = launch,
    #c=np.random.rand(3,1),
    marker = '.')
    #s =8) and use ax.scatter to plot points
    
#Experimental
ax.plot(posxData, 
posyData,
poszData, 
label = "EXP",
#c=np.random.rand(3,1),
marker = 'x')

#Making landing data text box and marking trajectories    
for launch in launches:
    xLand = np.round(launchData[launch]['posx'][-1],2)
    yLand = np.round(launchData[launch]['posy'][-1],2)
    lNum = launchData[launch]['num'][0]
    landInfo[lNum] = ('Launch '+str(lNum)+'\n'
        +str(launch) + '\n('
        +str(xLand) +', ' + str(yLand) +')' + '\n-----\n')
    ax.text(xLand,yLand,floor," "+str(lNum))    #Places a label at the landing position
    
#A very annoying amount of work to put the landing information in order
boxConstructor = 1
textBox2 = "Landings:\n---------\n"
for launch in launches:
    textBox2 +=landInfo[boxConstructor]
    boxConstructor +=1

#Plot formatting
ax.set_xlabel(niceNames[Axis[0]],fontdict=font)
ax.set_ylabel(niceNames[Axis[1]],fontdict=font)
ax.set_zlabel(niceNames[Axis[2]],fontdict=font)
ax.set_title('Trajectory Plot',fontdict=font,backgroundcolor='w',fontsize = 24)
ax.legend()
ax.text2D(0.05, 0.5, textBox1,backgroundcolor='w', transform=ax.transAxes)
ax.text2D(1.005, 0, textBox2,backgroundcolor='w', transform=ax.transAxes)
ax.text(0,0,0,"   launch")
ax.set_xlim(xMin,xMax)
ax.set_ylim(yMin,yMax)
ax.set_zlim(zMin,zMax)
ax.view_init(camAngle1,camAngle2)
plt.show()
fig.savefig('fig'+datafile+'.png',dpi=300)
