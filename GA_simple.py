import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ufl
from Homogenization.homogenization_simple import microstructure
from Optimization_Problems.plate_with_hole import solveMacroProblem
from Optimization_Problems.plate_with_hole import generate_mesh

def GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,func,des,w1,w2):
    readapt=0
    cost=np.ones((S,1))*1000
    prev_cost = None
    Pi=np.empty((0,S))
    meanParents=[]
    Orig=np.zeros((G,S))
    Children=np.zeros((K,dv))
    Parents=np.zeros((P,dv))
    Lambda=np.zeros((S,dv))
    Gen=1
    start = 0
    
    #Generate mesh
    gdim,fdim,domain, _ , facets =generate_mesh()
    #Generate starting population
    pop_new=np.random.uniform(lb, ub, (S, dv))
    

    while np.abs(np.min(cost))>TOL and Gen<G:
        print("**********************************")
        print("Generation number : ", Gen)
        pop=pop_new

        #Evaluate population fitness
        for i in range(start,S):
            print("String : ",i+1)
            cost[i]=func(pop[i,:],gdim,fdim,domain,facets,des,w1,w2)

        #Sort population fitnesses
        Index=np.argsort(cost[:,0])
        pop=pop[Index,:]
        cost=cost[Index,:]

        if readapt==1:
        # Readapting search interval 
            if prev_cost is not None and np.isclose(prev_cost, cost[0], atol=1e-5):
                lb = pop[0,:]- 0.3 * pop[0,:]
                print(lb)
                ub = pop[0,:] + 0.3 * pop[0,:]
                print(ub)

            prev_cost = cost[0]

        print(f"Best cost for generation {Gen} : {cost[0]}")

        #Select parents
        Parents=pop[0:P,:]
        meanParents.append(np.mean(cost[0:P]))

        #Generate K offspring
        for i in range(0,K,2):
            #Breeding parents
            alpha=np.random.uniform(0,1)
            beta=np.random.uniform(0,1)
            Children[i,:]=Parents[i,:]*alpha+Parents[i+1,:]*(1-alpha)
            Children[i+1,:]=Parents[i,:]*beta+Parents[i+1,:]*(1-beta)

        #Overwrite population with P parents, K children, and S-P-K random values
        pop_new=np.vstack((Parents,Children,np.random.uniform(lb, ub, (S-P-K, dv))))

        #Store costs and indices for each generation
        Pi= np.vstack((Pi, cost.T))
        Orig[Gen,:]=Index
        #Increment generation counter
        Gen=Gen+1
        #
        start = P

    #Store best population 
    Lambda=pop    
    meanPi=np.mean(Pi,axis=1)
    minPi=np.min(Pi,axis=1)
    return Lambda, Pi, Orig, meanPi, minPi,meanParents,cost

def CalcCost(vars,gdim,fdim,domain,facets,des,w1,w2):
    micro = microstructure(1e-3,120)
    micro.setGeomParam(vars[0],vars[1],vars[4],vars[5], vars[2], vars[3],vars[6])
    micro.setVoidRatio(vars[12],vars[9])
    micro.setSoftMaterial(vars[7],vars[8])    
    micro.setStiffMaterial(vars[10],vars[11])
    #micro.setStiffMaterial(110e9,0.31)
    #micro.setSoftMaterial(68e9,0.32)
    micro.generateRVE()
    C = micro.getHomogenizedProperties()
    #print(C)
    #print(C)
    #E2 = 68e9
    #nu2 = 0.32
    #C2 = (E2/(1-nu2**2))*np.array([[1,nu2,0],[nu2,1,0],[0,0,1-nu2]])
    if np.all(np.isnan(C)):
        #Set C to the softest material using an isotropic representation
        print("invalid stiffness from homogenization")
        micro.visualizeMicrostructure()
        #Set C to the isotropic representation of the softest material
        E_min = 68e9
        nu_min = .32
        C = (E_min/(1-nu_min**2))*np.array([[1,nu_min,0],[nu_min,1,0],[0,0,1-nu_min]])
        
    
    cost=  solveMacroProblem(ufl.as_matrix(C),gdim,fdim,domain,facets,w1,w2)
    print(cost)
    return cost



#Define genetic algorithm parameters
S=30
P=10
K=10
G=10
TOL=1e-3

#Number of design variables

# Geometric Parameters:
# Center (2) : x,y
# Exponents (2): P1, P2
# Radii (2): R1, R2
# Rotation (1): theta
# Material (6): E_soft, V_soft, void_ratio_soft, E_stiff, E_stiff, void_ratio_stiff

dv=13

#Upper and lower bounds for each variable

# Lower bounds
lb = np.array([-.5e-3,-1e-3,1,1,.1e-3,.1e-3,0,68e9,.32,0.0,340e9,0.27,0.0])

# Upper bounds
ub = np.array([.5e-3,1e-3,20,20,.8e-3,.8e-3,2*np.pi,88.5e9,.36,0.01,405e9,0.29,0.01])

#Desired designs and weights
Des1=0
Des2=0
w1=1e3
w2=1
des=[Des1,Des2]

#Call genetic algorithm to mininmize cost function
Lambda,Pi,Orig,meanPi,minPi,meanParents,costs=GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,CalcCost,des,w1,w2)

print("***************************************************************")
print("The design variables are:")
print(Lambda)
print("***************************************************************")
print("The corresponding costs are :")
print(costs)

#Plot results
plt.figure()
plt.semilogy(np.arange(len(meanPi)), meanPi, label="Mean cost")
plt.semilogy(np.arange(len(minPi)), minPi, label="Min cost")
plt.semilogy(np.arange(len(meanParents)), meanParents, label="Mean parent cost")
plt.title("Cost Evolution per Generation for Cost",fontsize=22)
plt.xlabel('Generation',fontsize=18)
plt.ylabel('Value',fontsize=18)
plt.legend()
plt.show()