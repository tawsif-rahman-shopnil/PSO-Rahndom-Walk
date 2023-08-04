import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import UnivariateSpline
from matplotlib import interactive


class CrtModel:
    def __init__(self):
        self.xs = -9
        self.ys = -17
        self.xt = 5
        self.yt = 17
        self.n = 3
        self.xobs = [1.5, 10, -13, -5, -14, 7, 4, -7.5, -2, 13, 15, 0.5, -12, 14, 8]
        self.yobs = [4.5, 3, .5, -5, -8, 8, -4, 5, 13, -11, 11, -14, 12, -2, 14]
        self.robs = [3.0, 2.0, 2.5, 4, 3.5, 2, 4, 3, 2.5, 3, 2.5, 2, 4, 2.5, 2]
        self.xmin = -20
        self.xmax = 20
        self.ymin = -20
        self.ymax = 20
    def cal_theta(self):
        return np.linspace(0,2*np.pi,100)


def generate_random_walk(steps, step_size, boundaries):
    current_position = np.random.uniform(boundaries[0], boundaries[1])
    random_walk = [current_position]

    for _ in range(steps):
        step = np.random.uniform(-step_size, step_size)
        current_position += step
        current_position = np.clip(current_position, boundaries[0], boundaries[1])
        random_walk.append(current_position)

    return random_walk


def analyze_random_walk(random_walk):
    mean = np.mean(random_walk)
    variance = np.var(random_walk)
    std_deviation = np.std(random_walk)

    return mean, variance, std_deviation


# Example usage
random_walk = generate_random_walk(steps=100, step_size=0.1, boundaries=[-20, 20])


class Solution:
    def __init__(self, mod):
        self.xmin=mod.xmin
        self.ymin=mod.ymin
        self.xmax=mod.xmax
        self.ymax=mod.ymax
        self.n=mod.n

    def CrtRandSol(self):
        self.x= np.random.uniform(self.xmin, self.xmax, self.n)
        self.y = np.random.uniform(self.ymin,self.ymax, self.n)


class MyCost:

    def __init__(self, mod):
        self.xs=mod.xs
        self.ys=mod.ys
        self.xt=mod.xt
        self.yt=mod.yt
        self.xobs=mod.xobs
        self.yobs=mod.yobs
        self.robs=mod.robs

    def ParseSol(self, x, y):

        self.XS=np.array([self.xs])
        self.XS=np.append([[self.XS]],[x])
        self.a=np.array([self.xt])
        self.XS=np.append([[self.XS]],[[self.a]])

        self.YS = np.array([self.ys])
        self.YS = np.append([[self.YS]], [y])
        self.a = np.array([self.yt])
        self.YS = np.append([[self.YS]], [[self.a]])

        self.k=self.XS.size
        self.TS=np.linspace(0,1, self.k)

        self.tt=np.linspace(0,1,100)
        splinex = UnivariateSpline(self.TS, self.XS)
        spliney = UnivariateSpline(self.TS, self.YS)
        splinex.set_smoothing_factor(0.2)
        spliney.set_smoothing_factor(0.2)
        self.xx = splinex(self.tt)
        self.yy = spliney(self.tt)
        self.dx=np.diff(self.xx)
        self.dy=np.diff(self.yy)
        self.L=np.sum(np.sqrt(self.dx**2+self.dy**2))
        nobs=len(self.xobs)
        self.Violation=0

        i=0
        while i<nobs :
            p= self.xx-self.xobs[i]
            q= self.yy-self.yobs[i]
            d=np.sqrt((p**2)+(q**2))
            v=np.maximum((1-d/self.robs[i]),0)
            self.Violation=self.Violation+ v.mean(axis=0)
            i+=1
        return self.L * (1 + 100*self.Violation)
        # End of Cost Function and Parse Sol

#Generating Model with three object
model=CrtModel()

# Generating Random Solution
# Generating Random Solution
#Sol1=Solution(model)
#Sol1.CrtRandSol()
#Initializing Cost Function
#Sol=MyCost( model)

#Basic PSO
nVar=model.n
VarSize=[1,nVar]

VarMin={}
VarMax={}
VarMin['x']=model.xmin
VarMin['y']=model.ymin
VarMax['x']=model.xmax
VarMax['y']=model.ymax

MaxIt=500
nPop=80
wmax=0.95
wmin=0.2
c1=2.0
c2=1.5

#Velocity parameter
alpha=0.1
VelMin={}
VelMax={}
VelMax['x']=alpha*(VarMax['x']-VarMin['x'])
VelMin['x']=-VelMax['x']
VelMax['y']=alpha*(VarMax['y']-VarMin['y'])
VelMin['y']=-VelMax['y']

#Particle Initialization
empty_Particle={
        'position':{},
        'velocity':{},
        'cost':{},
        'best_pos': {},
        'best_Sol':{},
        'best_cost': {},
        'violation':None
    }
#initialize Global Best
gbest={
    'pos':{},
    'cost': np.inf,
    'Sol' : {}
}

#Create Population
pop=[]

for i in range(0, nPop):
    pop.append(copy.deepcopy(empty_Particle))
    if i>0:
        randSol = Solution(model)
        randSol.CrtRandSol()
        pop[i]['position']['x'] = randSol.x.copy()
        pop[i]['position']['y'] = randSol.y.copy()
    else:
        xx = np.linspace(model.xs, model.xt, model.n + 2)
        yy = np.linspace(model.ys, model.yt, model.n + 2)
        pop[i]['position']['x'] = xx[1:len(xx) - 1]
        pop[i]['position']['y'] = yy[1:len(yy) - 1]
    #Initialize velocity
    pop[i]['velocity']['x'] = np.zeros(nVar)
    pop[i]['velocity']['y'] = np.zeros(nVar)
    #Evaluation
    Sol = MyCost(model)
    pop[i]['cost'] = Sol.ParseSol(pop[i]['position']['x'], pop[i]['position']['y'])
    pop[i]['best_Sol']['XS'] = Sol.XS
    pop[i]['best_Sol']['YS'] = Sol.YS
    pop[i]['best_Sol']['xx'] = Sol.xx
    pop[i]['best_Sol']['yy'] = Sol.yy
    #Update personal best
    pop[i]['best_pos']['x'] = pop[i]['position']['x']
    pop[i]['best_pos']['y'] = pop[i]['position']['y']
    pop[i]['best_cost'] = pop[i]['cost']
    pop[i]['violation'] = Sol.Violation
    if pop[i]['best_cost']<gbest['cost']:
        gbest['cost'] = pop[i]['best_cost']
        gbest['pos']['x'] = pop[i]['best_pos']['x']
        gbest['pos']['y'] = pop[i]['best_pos']['y']
        gbest['Sol']['XS'] = pop[i]['best_Sol']['XS']
        gbest['Sol']['YS'] = pop[i]['best_Sol']['YS']
        gbest['Sol']['xx'] = pop[i]['best_Sol']['xx']
        gbest['Sol']['yy'] = pop[i]['best_Sol']['yy']


BestCost = np.zeros(MaxIt)
Violation=0

#Animation parameter
axis={
    'x':None,
    'y':None
}

plot=[]

#Main PSO Loop

for it in range(0, MaxIt):
    w = wmax - ((wmax - wmin) *(pow((it/ MaxIt),alpha)))
    random_walk = generate_random_walk(steps=100, step_size=0.1, boundaries=[-20, 20])
    analyze_random_walk(random_walk)
    # Analyze random walk after PSO iterations
    if it == MaxIt - 1:
        mean, variance, std_deviation = analyze_random_walk(random_walk)
        print(f"Mean: {mean}")
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {std_deviation}")
    for i in range (0, nPop):
        #X Part
        #Update Velocity

        pop[i]['velocity']['x']=w*pop[i]['velocity']['x']+c1*np.random.rand(1,nVar)\
                                *(pop[i]['best_pos']['x']-pop[i]['position']['x'])\
                                + c2*np.random.rand(1,nVar)\
                                *(gbest['pos']['x']-pop[i]['position']['x'])

        #Update Velocity Bound
        pop[i]['velocity']['x'] = np.maximum(pop[i]['velocity']['x'], VelMin['x'])
        pop[i]['velocity']['x'] = np.minimum(pop[i]['velocity']['x'], VelMax['x'])

        #Update Position
        pop[i]['position']['x']=pop[i]['position']['x']+ pop[i]['velocity']['x']
        #Velocity Mirroring is not clear
        #Update Positon Bound

        pop[i]['position']['x'] = np.maximum(pop[i]['position']['x'], VarMin['x'])
        pop[i]['position']['x'] = np.minimum(pop[i]['position']['x'], VarMax['x'])



        #Y Part
        # Update Velocity
        pop[i]['velocity']['y'] = w * pop[i]['velocity']['y'] + c1 * np.random.rand(1,nVar)\
                                  * (pop[i]['best_pos']['y']- pop[i]['position']['y']) \
                                  + c2 * np.random.rand(1,nVar)\
                                  * (gbest['pos']['y'] - pop[i]['position']['y'])

        # Update Velocity Bound
        pop[i]['velocity']['y'] = np.maximum(pop[i]['velocity']['x'], VelMin['y'])
        pop[i]['velocity']['y'] = np.minimum(pop[i]['velocity']['x'], VelMax['y'])

        # Update Position
        pop[i]['position']['y'] = pop[i]['position']['y'] + pop[i]['velocity']['y']

        # Update Positon Bound
        pop[i]['position']['y'] = np.maximum(pop[i]['position']['y'], VarMin['y'])
        pop[i]['position']['y'] = np.minimum(pop[i]['position']['y'], VarMax['y'])

        # Evaluation
        Sol = MyCost(model)
        pop[i]['cost'] = Sol.ParseSol(pop[i]['position']['x'], pop[i]['position']['y'])
        pop[i]['violation'] = Sol.Violation
        #Update Personal Best
        if pop[i]['cost'] <pop[i]['best_cost']:
            pop[i]['best_Sol']['XS'] = Sol.XS.copy()
            pop[i]['best_Sol']['YS'] = Sol.YS.copy()
            pop[i]['best_Sol']['xx'] = Sol.xx.copy()
            pop[i]['best_Sol']['yy'] = Sol.yy.copy()
            # Update personal best
            pop[i]['best_pos']['x'] = pop[i]['position']['x']
            pop[i]['best_pos']['y'] = pop[i]['position']['y']
            pop[i]['best_cost'] = pop[i]['cost'].copy()

            #Update Global Best
            if pop[i]['best_cost'] < gbest['cost']:
                gbest['cost'] = pop[i]['best_cost']
                gbest['pos']['x'] = pop[i]['best_pos']['x']
                gbest['pos']['y'] = pop[i]['best_pos']['y']
                gbest['Sol']['XS'] = pop[i]['best_Sol']['XS']
                gbest['Sol']['YS'] = pop[i]['best_Sol']['YS']
                gbest['Sol']['xx'] = pop[i]['best_Sol']['xx']
                gbest['Sol']['yy'] = pop[i]['best_Sol']['yy']

        Violation = int(Sol.Violation)
    #Update best cost ever found
    BestCost[it]= gbest['cost']

    #Storing data frame for animation
    plot.append(copy.deepcopy(axis))
    plot[it]['x']=gbest['Sol']['xx']
    plot[it]['y']=gbest['Sol']['yy']

    #Intertia Weight Dumping
    #w*=wdamp

    if Violation>0:
        Flag='Violation'+ str(Violation)
    else:
        Flag='*'

    print("Iteration", it, "Best Cost", BestCost[it], Flag)



fig = plt.figure(1)
ax = plt.axes(xlim=(model.xmin, model.xmax), ylim=(model.ymin, model.ymax))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    theta = model.cal_theta()
    for i in range(len(model.xobs)):
        plt.fill(model.xobs[i] + model.robs[i] * np.cos(theta), model.yobs[i] + model.robs[i] * np.sin(theta))
    plt.plot(model.xs, model.ys, '*')
    plt.plot(model.xt, model.yt, '*')
    return line,

# animation function. This is called sequentially
def animate(i):
    x = plot[i]['x']
    y = plot[i]['y']
    line.set_data(x, y)
    return line,

fig = plt.figure(1)
ax = plt.axes(xlim=(model.xmin, model.xmax), ylim=(model.ymin, model.ymax))
line, = ax.plot([], [], lw=2)

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=40, interval=20, repeat= False)

plt.show()
plt.figure(2)

plt.plot(BestCost)
plt.show()

plt.plot(random_walk)
plt.xlabel('Step')
plt.ylabel('Position')
plt.title('Random Walk')
plt.show()






#Printing Graph
theta=model.cal_theta()

for i in range(len(model.xobs)):
    plt.fill(model.xobs[i]+model.robs[i]*np.cos(theta), model.yobs[i] + model.robs[i] * np.sin(theta) )


plt.plot(gbest['Sol']['XS'], gbest['Sol']['YS'], 'x', 'LineWidth', 2)
plt.plot(gbest['Sol']['xx'], gbest['Sol']['yy'])
plt.plot(model.xs, model.ys, '*')
plt.plot(model.xt, model.yt, '*')
plt.show()
    #plt.pause(0.1)
    #plt.close()

