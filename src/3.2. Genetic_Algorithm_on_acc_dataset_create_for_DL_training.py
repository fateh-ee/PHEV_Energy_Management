import random
import matplotlib.pyplot as plt
import numpy as np


def fitness_function(solution,SOC): # solution = set of solutions of all segments in a driving cycle, SOC= initial  value before starting the simulation

    cost_all=[] #to store cost of each segment of driving cycle
    
    for i in range(len(solution)): #to consider every segment serially (of driving cycle) for performing simulation
        
        #solution of one segment
        acc=acc_segment[i] # considering specific segment of the driving cycle
        solution_seg = solution[i] # considering specific solution corresponding to that particular segment
        
        fitness= fitness_function_main(solution_seg,SOC,acc) # calculating fitness for that particular segment
        
        if fitness[0]==0: #if the solution results: SoC < 0.3 (after calculating fitness/running simulation) then returning fitness=0 means worst fitness minimum
            return 0,0
    
        cost_all.append(fitness) #adding cost of each segment
        
        SOC=fitness[1]#updating soc after running simulation of each segment


        #****************************** this part is added for storing input dataset of every segment of one driving cycle for ML training
        # storing input data = x (acceleration: summation , mean , standard_deviation , SoC_current )
        if i!=0: # because, we store upcomming output/solution for previous acceleration segment. like this, sol[i=1] for acc[i=0] but if sol[0] then there is no acc[-1]
            # x: acc[i-1] (total,mean,std), soc[i], y: sol[i] for acc[i]
            seg_current=[np.sum(acc_segment[:i]),np.mean(acc_segment[:i]),np.std(acc_segment[:i]),SOC] # here input acceleration data is not only for current segment but also the previous all segments so that DL model can analyze all the previous segments to predict the upcomming segment. 
            # for example: if output= solution_for_acc_seg_3, then input= considers, (acc_seg_3 + acc_seg_2 + acc_seg_1 + acc_seg_0)
            x_current.append(seg_current)
        #******************************
        

    cost_final=sum((x[0]) for x in cost_all)-sum(x[1] for x in cost_all )*pel + SOC*pel # removing the value of soc of each segment to consider only the final value of soc at the end of last segment of the driving cycle   
    
    return 1/cost_final, SOC # inverse of cost_final is performed to make the fitness value higher with minimum cost so that the solution set with higher fitness value will have higher priority during selecting best solution by GA
    # here returning, SOC = final value of soc at the end of the driving cycle after performing simulation on all segments


# 2. second part (main) which will perform simulation on every particular segment of one driving cycle
def fitness_function_main(solution,SOC,acc): 
    BK1, CL1, CL2, Te, we, Tm, Tg = solution # 7 control actions
    Tm_max, Tg_max, Te_max, we_max = 350, 200, 200, 4500 # those are highest constraints. GA randomly adjust Te,we,Tm,Tg to meet closer acceleration requirement. 
    const_acc=0.009 # acc constant to consider it's affect on soc charging/discharging and cost function
    J=0 # initial cost at the begining of the simulation
  
    # Mode 1 : Electric Vehicle Mode 1 (EV1): only motor runs
    if (BK1,CL1,CL2)==(1,0,0): 
        for a in acc: # acc = acceleration dataset of one particular segment of the whole driving cycle. a = acceleration at every second on that segment
            if a > 0: # increasing acceleration
                SOC-=(a*const_acc) # SOC discharging rate
                J+=abs(a-Tm/(Tm_max/max(acc_full)))*0.01 # special term: Tm, only motor is supplying power. special term is used to meet closer torque value according to that acceleration segment. which mathematically effects like this example: if, (Tm > Tm_required) or (Tm < Tm_required) then cost increases. if, Tm is closer to required then cost will be lower. In the equation, 0.01 is just a constant
            else:
                SOC+=0 # decreasing acc and no regenerative braking works, so SOC will be unchanged
                
            if SOC < 0.3: # while it results: SOC < 0.3 , then making fitness= 0 to make lowest/worst fitness value to avoid this solution by GA
                return 0,0
            if SOC > 0.9: # maximum threshold value of SOC
                SOC=0.9
            
        J+= SOC * pel # considering final value of SOC at the end of one segment
            
    # Mode 2 : Electric Vehicle Mode 2 (EV2): both motor and generator (during regenerative braking) runs
    elif (BK1,CL1,CL2)==(0,0,1):      
        for a in acc: # acc = acceleration dataset of one particular segment of the whole driving cycle. a = acceleration at every second on that segment
            
            if a > 0: # increasing acceleration
                SOC-=(a*const_acc) # SOC discharging rate
                J+=abs(a-Tm/(Tm_max/max(acc_full)))*0.01 # Tm, only motor is supplying power. special term is for meeting closer torque value according to that acceleration segment. which mathematically effects like this example: if, (Tm > required) or (Tm < required) then cost increases. if, Tm is closer to required then cost will be lower. In the equation, 0.01 is just a constant
            
            else: # decreasing acceleration
                SOC+=(a*const_acc)*(-1)  # Regenerative braking charges battery. Here, '-1' is multiplied to consider the affect of charging soc (to increase soc value) during negative acceleration
                
                J+=abs(a-Tg/(Tg_max/max(acc_full)))*0.01 # special term: torque of generator, Tg during regenerative braking. special term is for meeting closer torque value according to that acceleration segment. which mathematically effects like this example: if, (Tg > required) or (Tg < required) then cost increases. if, Tg is closer to required then cost will be lower. In the equation, 0.01 is just a constant
            
            if SOC < 0.3: # while it results: SOC<0.3 , then making fitness= 0 to make lowest fitness value to avoid this solution by GA
                return 0,0
            if SOC > 0.9: # maximum threshold value of SOC
                SOC=0.9
                                                                
        J+= SOC * pel # considering final value of SOC at the end of one segment
        
    # Mode 3 : Range Extender Mode (REV): Engine will charges the vehicle as well as charges battery. regenerative braking does not work 
    elif (BK1,CL1,CL2)==(1,1,0): 
        for a in acc: # acc = acceleration dataset of one particular segment of the whole driving cycle. a = acceleration at every second on that segment
            if a > 0: # increasing acceleration
                SOC+=(a*const_acc) # SOC discharging rate
                J+=pfu*fpu*a+abs(a-Te/(Te_max/max(acc_full)))*0.01+abs(a-we/(we_max/max(acc_full)))*0.01 # cost function considering fuel cost. special term: torque of and rotation speed of engine, Te and we during engine running condition. special term is for meeting closer torque value according to that acceleration segment. which mathematically effects like this example: if, (Te or we > required) or (Te or we < required) then cost increases. if, Te or we is closer to required then cost will be lower. In the equation, 0.01 is just a constant
            else:  # decreasing acceleration
                SOC+=0.00 # decreasing acc and no regenerative braking works, so SOC will be unchanged
            if SOC < 0.3: # while it results: SOC<0.3 , then making fitness= 0 to make lowest fitness value to avoid this solution by GA
                return 0,0
            if SOC > 0.9:  # maximum threshold value of SOC
                SOC=0.9
             
        J+= SOC * pel  # considering final value of SOC at the end of one segment
        
    # Mode 4 : Hybrid Electric Vehicle Mode (HEV): Both motor and engine will run the vehicle with 30% and 70% power sharing respectively. Generator (regenerative braking) will also work 
    else: #(0,1,1)
        for a in acc: # acc = acceleration dataset of one particular segment of the whole driving cycle. a = acceleration at every second on that segment
            if a > 0: # increasing acceleration
                # here motor is providing 30% of the required power which saves engine fuel by 30%
                SOC-=(a*const_acc)*0.30 # SOC discharging by 30%
                J+=pfu*fpu*a*0.70 + abs(a-Te/(Te_max/max(acc_full)))*0.01*0.70 + abs(a-we/(we_max/max(acc_full)))*0.01* 0.70 + abs(a-Tm/(Tm_max/max(acc_full)))*0.01* 0.30 # both motor and engine supply power by a percentage sharing so, Tm,Te,we terms exist            
            else: # decreasing acceleration
                SOC+=(a*const_acc)*(-1) # Regenerative braking charges battery. Here, '-1' is multiplied to consider the affect of charging soc (to increase soc value) during negative acceleration
                J+=abs(a-Tg/(Tg_max/max(acc_full)))*0.01  # special term: torque of generator, Tg during regenerative braking. special term is for meeting closer torque value according to that acceleration segment. which mathematically effects like this example: if, (Tg > required) or (Tg < required) then cost increases. if, Tg is closer to required then cost will be lower. In the equation, 0.01 is just a constant
            if SOC < 0.3: # while simulation results: SOC<0.3 , then making fitness= 0 to make lowest fitness value to avoid this solution by GA
                return 0,0
            if SOC > 0.9:  # maximum threshold value of SOC
                SOC=0.9
            
        J+=SOC * pel # considering final value of SOC at the end of one segment
    return J,SOC # returning cost and soc value (after calculating fitness of one particular segment) to the first part of fitness function (for the specific segment of whole driving cycle)



# Genetic Algorithm
#1. Initialize population = number of solutions = number of chromosomes , each chromosome consist of genes , each gene has 7 element / control actions 
# number of genes = number of segment of the driving cycle with specific segment size
# every gene has 7 elements / 7 control actions
# Example: Solution_1 = [ gene_1 , gene_2 , .. gene_n ] = [ [1,0,0,2,43,-2,4] , [0,0,1,4,21,5,-2] , .. [1,1,0,56,-43,34,10] ] of one driving cycle with the segment size 'n'
def initialize_population(population_size): # this function creates number of initial solutions as population according to given population size, after that GA is performed on that solution.

    population = []
    for _ in range(population_size):
        solution=[]
        for _ in range(segment):# each solution consists of few (given) segments of one overall driving cycle
            (BK1,CL1,CL2)=random.choice([(1,0,0),(0,0,1),(1,1,0),(0,1,1)]) # to select among four modes of operation
            if (BK1,CL1,CL2) == (1,0,0): # Mode1: EV1: only motor runs
                Te  = 0
                we  = 0
                Tm  = random.randint(-350, 350) # motor torque (Nm)
                Tg  = 0
            if (BK1,CL1,CL2) == (0,0,1): # Mode2: EV2: both motor and generator run
                Te  = 0
                we  = 0
                Tm  = random.randint(-350, 350) #Nm
                Tg  = random.randint(-200, 200) #generator torque (Nm)
            if (BK1,CL1,CL2) == (1,1,0): # Mode3: REV: Engine runs
                Te  = random.randint(0, 200) #engine torque (Nm)
                we  = random.randint(0, 4500) #engine rotational speed (r/min)
                Tm  = 0
                Tg  = 0
            if (BK1,CL1,CL2) == (0,1,1): # Mode4: engine (70% of power sharing), motor (30% of power sharing) and generator (regenarative braking) runs
                Te  = random.randint(0, 200) #Nm
                we  = random.randint(0, 4500) #r/min
                Tm  = random.randint(-350, 350) #Nm
                Tg  = random.randint(-200, 200) #Nm
            solution.append([BK1, CL1, CL2, Te, we, Tm, Tg]) # one driving cycle = [solution_for_segment_1 , solution_for_segment_2, ... solution_for_segment_n ] = one solution (set of solutions in one driving cycle according to segment size 'n' ). appending solution (consists of 7 elements/ control actions) at each segment (each gene) of one driving cycle
        population.append(solution) # appending the number of solutions according to a given number. each solution is for one whole driving cycle which is a set of solutions according to given segment size of that driving cycle
    return population


#2. selection
def selection(population): # selection of top 10 solutions, parent 1 group, parent 2 group of solutions
    population.sort(key=lambda x: fitness_function(x, SOC)[0], reverse=True) # Sort population based on highest fitnes values (descending order)
    
    top_10 = population[:10] # Select top 10 individuals
    top_120 = population[:120] # Select top 120 individuals
    
    parent1 = top_120[:60] # top 60 as parents_group_1
    parent2 = top_120[60:] # remaining 60 as parents_group_2 
    
    return parent1, parent2, top_10


def crossover(parent_1, parent_2): # crossover between two genes of two parents
    
    crossover_point = random.randint(1, len(parent_1) - 1) # Perform crossover operation by taking a random crossover point to swap values
    
    # Define valid patterns and their corresponding replacements.
    patterns = {  # to make sure that first 3 elements should be in between 4 modes of operation
        (0, 0, 0): (0, 0, 1), 
        (0, 1, 0): (0, 1, 1),
        (1, 0, 1): (1, 0, 0), # if one single gene of a child has first three element (1,0,1) then we try to convert it to the closest possible mode (1,0,0)
        (1, 1, 1): (1, 1, 0)
    }
    
    # Defining crossover function for a gene of a single child
    def crossover_child(parent_a, parent_b):
        child = parent_a[:crossover_point] + parent_b[crossover_point:]
        pattern = tuple(child[:3]) # converting to possible combination by avoiding invalid combination 
        if pattern in patterns: # if the combination of first three element of a gene is invalid
            child[:3] = patterns[pattern] # then we try to convert it into closest possible valid combination
        return child # returning the gene of the particular segment of the child
        
    
    # Generate children using crossover function. One crossover (between two gene of two parents) creates two genes of two children
    child_1 = crossover_child(parent_1, parent_2) # first gene of child 
    child_2 = crossover_child(parent_2, parent_1) # second gene of child
    return child_1, child_2 # returning two genes of the child



# 4. Mutation
# 4.1 preparing 2 groups of solutions for applying mutation (first group: mutation will be applied, second group: mutation wont be applied ) 
def apply_mutation(children, mutation_rate):
    
    mutated_children = [] 
    non_mutated_children = []
    
    # Randomly select 10 children for mutation
    random_10_indices = random.sample(range(len(children)), 10) # just creating 10 random numbers with the values from 1 to 120

    for i, child in enumerate(children):
         
        if i in random_10_indices: # if i- is among randomly selected children, then we mutate and add it to the group of mutated children
            mutated_child = mutation(child, mutation_rate)
            mutated_children.append(mutated_child)
        else:
            non_mutated_children.append(child) # if i- is not among randomly selected children then we dont mutate and add it to the group of non mutated children
     
    return mutated_children, non_mutated_children 


# 4.2 applying mutation
def mutation(solution, mutation_rate):
    mutated_solution =[] # to contain whole solution
    for i in range(len(solution)): # considering each gene/segment
        solution_seg=solution[i]  # select one gene
        mutated_solution_seg = [] # to contain 7 mutated element of a single gene ( one gene (/solution of one segment) is like this: [1,1,0,12,-23,5,76]) to the solution 
        
        
        # Mutate the first three elements of a single gene while ensuring they come from the predefined set
        mutated_solution_seg.extend(random.choice([(1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]))        
        
        # Mutate the other four element of a single gene
        for gene in solution_seg[3:]:
            mutated_gene = gene * random.uniform(1 - mutation_rate / 200, 1 + mutation_rate / 200)
            mutated_solution_seg.append(round(mutated_gene)) # [ 1,1,0,. storing 7 mutated element of a single gene ]
        
        mutated_solution.append(mutated_solution_seg) # [[1,1,0,12,-23,5,76],[] .. storing every gene (one solution of one segment) to make a whole mutated solution of one driving cycle]
    
    return mutated_solution # return whole mutated solution which is a set of solutions according to the predefined segments of one driving cycle



# Executing genetic algorithm
def genetic_algorithm(population_size, mutation_rate, generations):
    population = initialize_population(population_size) #initialize population

    for i in range(generations): # specified generations/iterations
        parent_1, parent_2, top_10 = selection(population) # selection of parent_1 (60 solutions), parent_2 (60 solutions), top_10 (top 10 best solutions of one population)
        
        # Applying crossover to each pair of parents     
        children = [] 

        for solution1, solution2 in zip(parent_1, parent_2): # to access each solution/chromosome of parent_1 and parent_2 respectively
            child_1_solution = []
            child_2_solution = []
            for gene1, gene2 in zip(solution1, solution2): # to access each genes/seven_elements in solution_1 and solution_2 serially
                
                # we perform crossover between two genes of two parents serially.  
                child_1_gene, child_2_gene = crossover(gene1, gene2) # crossover between two parent genes and create 2 genes of 2 children
                
                # appending gene on child_1 and child_2 serially after crossover
                child_1_solution.append(child_1_gene) # first gene goes to child_1 with the same serial after crossover
                child_2_solution.append(child_2_gene) # second gene goes to child_2 with the same serial after crossover
                
            children.extend([child_1_solution, child_2_solution]) # creating group of 120 children
        # len(children) = 120 , after performing crossover between parents_1 (60 solutions) x parents_2 (60 solutions)             
        
        
        mutated_children, non_mutated_children = apply_mutation(children, mutation_rate) # applying mutation on the randomly selected 10 childrean and also getting the group non mutated children
        
        
        population = top_10 + mutated_children + non_mutated_children # 10 + 10 + 110 = 130 offspring is generated after the end of one generation

        best_solution = max(population, key=lambda x: fitness_function(x,SOC)) # selection of best solution with highest fitness value
        cost_function= fitness_function(best_solution,SOC) # cost_funtion = [ 1/(calculated_fitness_cost) , SOC_final ] 
        cost_value_current= round(1/cost_function[0],2) # real minimum cost value after inversing. because the inverse value is returned from fitness function to get the highest fitness value with minimum cost.
        
    
    return best_solution #returning top 1 best solution after the last generation





# Application
population_size = 130 # number of population = number of solutions = number of chromosomes = each chromosome consists of genes which are the segments of one driving cycle
generations = 50 # number of iterations = number of generations. initially 20 is selected to quickly see which segment size gives better solution of one unique driving cycle. later we apply GA with generation 50 with the best segment size to get more minimum cost
mutation_rate=2 # %(+-(percentage/ 2)). here mutation_rate = in between +1 to -1 , 

#fitness function value
pfu=0.795  # pu cost of fuel
pel=0.137 # pu cost of electricity
fpu=1 #fuel consumption for per unit acceleration
J=0 # initial cost 
SOC=0.9 #initial value of SOC

########################################################################
# loading acceleration data of a driving cycle
acc_full=np.loadtxt('acc_sc03.txt')
#acc_full=[0,0,1,2,0,0,3]

acc_size=len(acc_full)
print(f'acc size:{acc_size:} seconds')

chunks=[(i+1) for i in range(int((acc_size)/2))] 
segments=sum(int(acc_size/(i+1)) for i in range(acc_size))-acc_size
print(f'Possible chunks: {chunks:}')
print(f'Possible datasets for training: {segments:} ')

x_current=[] # to store current input data (acceleration: summation , mean , standard_deviation , SoC_current ) for a specific segment size
y=[] # to store all the current solutions for upcomming acc segments
x=[] # to store all the current input data for current acc segments
for i in chunks: #takes different chunks and create a segment
    chunk_size=i
    segment=int(acc_size/chunk_size)
    acc_segment = [acc_full [i*chunk_size:(i+1)*chunk_size] for i in range(segment)]
    
    
    current_solution=genetic_algorithm(population_size, mutation_rate, generations)
    
    x_current=[]
    cost=fitness_function(current_solution,SOC) # cost[0]= 1 / cost_real, cost[1]=soc
    
    #storing x and y
    x.extend(x_current) # storing current acceleration input (with the specific segment size)
    y.extend(current_solution[1:]) # storing upcomming solution (with the specific segment size)
    print(f'dataset: {len(x):}/{segments:}') # print: how many datasets has been created uptill now / total datasets
    
    #saving it to train our DL model
    np.savetxt('dc_sc03_x.txt',x)
    np.savetxt('dc_sc03_y.txt',y)



