from math import factorial
import random

import matplotlib.pyplot as plt
import numpy
import pymop.factory
import pandas as pd

from chromosome import Chromosome
from metrics import Metrics
from operators import Operators
from dataset import Dataset
from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

# Problem definition

toolbox = base.Toolbox()
data=   pd.read_csv("C:/Users/Jose/Desktop/TFG/data/FA(in).csv", header=0, sep=';')
df = pd.DataFrame(data)

toolbox.register("dataset", Dataset, dataset=df)
PROBLEM = "dtlz2"
NOBJ = 5
K = 10
NDIM = NOBJ + K - 1
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)
##

# Algorithm parameters
MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 1.0
MUTPB = 1.0
##

# Puntos de referencia para selNSGAIII
ref_points = tools.uniform_reference_points(NOBJ, P)

# Creadores clases necesarias para DEAP 
## Maximizamos: soporte, confianza, numero de atributos en regla de asociación (máximo 2 en antecedente y 2 en consecuente)
## Minimizamos: recov, amplitud media de los intervalos
creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0,-1.0,1.0,-1.0) )
creator.create("Individual", Chromosome, fitness=creator.FitnessMulti)

w= [1.0,1.0,-1.0,1.0,-1.0]
toolbox.register("individual", Chromosome.create_chromosome, dataset=toolbox.dataset())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: ind.chromosome_eval(toolbox.dataset(), w))
toolbox.register("mate", Operators.crossover, dataset=toolbox.dataset())
toolbox.register("mutate", Operators.mutation, dataset=toolbox.dataset())
toolbox.register("select", tools.selNSGA3,  ref_points=ref_points)



def main(seed=None):
    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, 3):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook


if __name__ == "__main__":
    pop, stats = main()
    pop_fit = numpy.array([ind.fitness.values for ind in pop])

    pf = problem.pareto_front(ref_points)
    print(igd(pop_fit, pf))

    # import matplotlib.pyplot as plt
    # import mpl_toolkits.mplot3d as Axes3d

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection="3d")

    # p = numpy.array([ind.fitness.values for ind in pop])
    # ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", s=24, label="Final Population")

    # ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], marker="x", c="k", s=32, label="Ideal Pareto Front")

    # ref_points = tools.uniform_reference_points(NOBJ, P)

    # ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o", s=24, label="Reference Points")

    # ax.view_init(elev=11, azim=-25)
    # ax.autoscale(tight=True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("nsga3.png")