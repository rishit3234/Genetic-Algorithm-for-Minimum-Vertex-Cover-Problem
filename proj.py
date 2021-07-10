import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from array import *
 
 
n = 40
graph = []
for i in range(n):
  node = []
  for j in range(n):
    p = random.uniform(0,1)
    if(p<0.5):
        node.append(1)
    else:
        node.append(0)
  graph.append(node)
 
for i in range(n):
  for j in range(0,i):
    graph[i][j] = graph[j][i]
 
for i in range(n):
  graph[i][i] = 0
 
 
G = nx.Graph()
 
Gen=np.array([])
Fit=np.array([])
def create_individual():
    individual=[]
    for i in range(n):
        individual.append(random.randint(0,1))
    return individual

 
'''Creating Population'''
population_size=50
generation=0
population=[]
for i in range(population_size):
    individual=create_individual()
    population.append(individual)
    
'''Fitness'''
def fitness(graph,individual):
    fitness=0
    for i in range(n):
        if individual[i]==1:
            fitness+=1
        for j in range(i):
            if graph[i][j]==1:
                if individual[i]==0 and individual[j]==0:
                    fitness+=10
    return fitness
 

'''Crossover'''
def crossover(parent1,parent2):
    position=random.randint(1,n-2)
    child1=[]
    child2=[]
    for i in range(position+1):
        child1.append(parent1[i])
        child2.append(parent2[i])
    for i in range(position+1,n):
        child1.append(parent2[i])
        child2.append(parent1[i])
    return child1,child2
 
'''Mutation'''
def mutation(individual,probability):
    check=random.uniform(0,1)
    if check<=probability:
        position=random.randint(0,n-1)
        individual[position]=1-individual[position]
    return individual
 
'''Tournament Selection'''
def tournament_selection(population):
    new_population=[]
    for j in range(2):
        random.shuffle(population)
        for i in range(0,population_size-1,2):
            if fitness(graph,population[i])<fitness(graph,population[i+1]):
                new_population.append(population[i])
            else:
                new_population.append(population[i+1])
    random.shuffle(new_population)
    return new_population
 
best_fitness=fitness(graph,population[0])
fittest_individual=population[0]
gen=0
while(gen!=1000):
    best_fitness=fitness(graph,population[0])
    fittest_individual=population[0]
    for individual in population:
        f=fitness(graph,individual)
        if f<best_fitness:
            best_fitness=f
            fittest_individual=individual
    if gen%100==0:
        print("Generation: ",gen,"Best Fitness: ",best_fitness,"Individual: ",fittest_individual)
    Gen=np.append(Gen,gen)
    Fit=np.append(Fit,best_fitness)
    gen+=1
    population=tournament_selection(population)
    new_population=[]
    for i in range(0,population_size-1,2):
        child1,child2=crossover(population[i],population[i+1])
        new_population.append(child1)
        new_population.append(child2)
    for individual in new_population:
        if(gen<400):
            individual=mutation(individual,0.4)
        else:
            individual=mutation(individual,0.2)
    population=new_population
    
    
    
print(fitness(graph,fittest_individual))

for i in range(n):
    G.add_node(i,val = fittest_individual[i])
    
color_map = nx.get_node_attributes(G, "val")

for i in range(n):
  for j in range(0,i):
    if graph[i][j] == 1:
      G.add_edge(i,j)
      
for key in color_map:
    if(color_map[key]==0):
        color_map[key]="blue"
    else:
        color_map[key]="red"

color_nodes = [color_map.get(node) for node in G.nodes()]
     
nx.draw(G, with_labels=True, node_color = color_nodes)
plt.show()
plt.title("Fitness vs Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.plot(Gen,Fit)
plt.show()

vis = []
ans = 0
for i in range(n):
    vis.append(0)

for i in range(n):
    if(vis[i] == 1):
        continue
    f = 0
    for j in range(n):
        if(graph[i][j] == 1):
            if(vis[j] == 0):
                ans = ans + 1
                vis[j] = 1

print(ans)
print(vis)

