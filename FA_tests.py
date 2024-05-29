from chromosome import Chromosome
from metrics import Metrics
from operators import Operators
from dataset import Dataset
import pandas as pd
import random
from deap import base, creator, tools, algorithms


def main():
    #### Datos para la comprobación del funcionamiento de las clases

    # DEAP Initialization
    

    # Toolbox para configurar los algoritmos genéticos
    toolbox = base.Toolbox()

    #df =   pd.read_csv("C:/Users/Jose/Desktop/TFG/data/FA(in).csv", header=0, sep=';')
    data = {
        'A': [1.2, 2.3, 5.6],
        'B': [7.8, 8.9, 3.3],
        'C': [9.1, 3.2, 4.8]
    }

    df = pd.DataFrame(data)

    w=1.

    toolbox.register("dataset", Dataset, dataset=df)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", Chromosome, fitness=creator.FitnessMax)

    toolbox.register("individual", Chromosome.create_chromosome, dataset=toolbox.dataset())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: ind.chromosome_eval(toolbox.dataset(), w))
    toolbox.register("mate", Operators.crossover)
    toolbox.register("mutate", Operators.mutation, dataset=toolbox.dataset())
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create the population
    population = toolbox.population(n=10)

    # # Poblacion inicial
    acum=0.
    for i, ind in enumerate(population):
        #print(f"Cromosoma numero {i}:")
        print(f"Intervalos: {ind.intervals}")
        print(f"Transacciones: {ind.transactions}")
        print(f"Soporte calculado: {ind.support}")

        #print(f"Fitness: {ind.fitness}\n")
        acum+= ind.fitness.values[0]
    print(acum)

    # Define the parameters for the eaSimple algorithm
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2

    # Run the evolutionary algorithm
    popfin, logBook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)
    print(logBook)
    acumf=0.
    ## Poblacion final    
    for i, ind in enumerate(popfin):
        # print(f"Cromosoma numero {i}:")
        print(f"Intervalos: {ind.intervals}")
        print(f"Transacciones: {ind.transactions}")
        print(f"Soporte calculado: {ind.support}")
        # print(f"Fitness: {ind.fitness}\n")
        # print(Metrics.fitness(ind, toolbox.dataset(), w))
        acumf+= ind.fitness.values[0]
    print(acumf)
    
if __name__ == "__main__":
    main()