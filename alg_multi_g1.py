from math import factorial
import random
import matplotlib.pyplot as plt
import numpy as np
import pymop.factory
from chromosome import Chromosome
from operators import Operators
from metrics import Metrics
from dataset import Dataset
from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools
from scoop import futures

# Definir el problema DTLZ4
PROBLEM = "dtlz4"
NOBJ = 2  # Número de objetivos
K = 8   # Número de variables relacionadas con los objetivos
NDIM = NOBJ + K - 1  # Número total de variables

problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)

# Parámetros del algoritmo
MU = 1000  # Tamaño de la población
NGEN = 100  # Número de generaciones
CXPB = 0.8  # Probabilidad de cruce
MUTPB = 0.1  # Probabilidad de mutación

# Crear puntos de referencia uniformes
ref_points = tools.uniform_reference_points(NOBJ, 12)

#### Datos para la comprobación del funcionamiento de las clases

# Inicialización DEAP

# Toolbox para configurar los algoritmos genéticos
toolbox = base.Toolbox()

toolbox.register("dataset", Dataset)


#print(toolbox.dataset().dataframe)
#print(toolbox.dataset().column_ranges)
creator.create("Fitness", base.Fitness, weights=(1.,1.))
creator.create("Individual", Chromosome, fitness=creator.Fitness)

toolbox.register("individual", Chromosome.create_chromosome)
#print(len(toolbox.individual().fitness.values))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Especificamos multi
Chromosome.MULTI = True
toolbox.register("evaluate", Chromosome.chromosome_eval_multi)
toolbox.register("mate", Operators.crossover)
toolbox.register("mutate", Operators.mutation)

pop   = toolbox.population(n=MU)

# Definir puntos de referencia para NSGA-III
#ref_points = tools.uniform_reference_points(nobj=2, p=120) # H=7, Nob=3 (11 sobre 3)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)


# Evalución paralela para agilizar calculos
toolbox.register("map", futures.map)

def main(seed=None):
    random.seed(seed)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_sup = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_lev = tools.Statistics(lambda ind: ind.fitness.values[1])

    stats_fit.register("avg", np.mean, axis=0)
    stats_fit.register("std", np.std, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)

    stats_sup.register("avg", np.mean)
    stats_sup.register("max", np.max)

    stats_lev.register("avg", np.mean)
    stats_lev.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, support=stats_sup, leverage=stats_lev)
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max", "avgSup", "maxSup", "avgLev", "bestLev"

    pop = toolbox.population(n=MU)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    best_ind = tools.selNSGA3(pop, 1, ref_points)[0]
    sup_best = best_ind.support[2]
    lev_best = Metrics.calculate_leverage(best_ind)

    record = mstats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), avg=np.round((record['fitness']['avg']),2), std=np.round((record['fitness']['std']),2),
                    min=np.round((record['fitness']['min']),2),
                    max=np.round((record['fitness']['max']),2),
                   avgSup=round(record['support']['avg'],2), bestSup=round(sup_best,2),
                   avgLev=round(record['leverage']['avg'],2), bestLev=round(lev_best,2)
                   )

    print(logbook.stream)


    hof_aprox =  tools.selNSGA3(pop, 100, ref_points)

    for gen in range(1, NGEN):
        offspring = toolbox.select(pop, MU - len(hof_aprox))
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        offspring.extend(hof_aprox)

        pop[:]= offspring
        hof_aprox =  tools.selNSGA3(pop, 100, ref_points)
        best_ind = tools.selNSGA3(pop, 1, ref_points)[0]
        sup_best = Metrics.calculate_lift(best_ind)
        lev_best = Metrics.calculate_gain(best_ind)

        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), avg=np.round((record['fitness']['avg']),2), std=np.round((record['fitness']['std']),2),
                        min=np.round((record['fitness']['min']),2),
                        max=np.round((record['fitness']['max']),2),
                    avgSup=round(record['support']['avg'],2), bestSup=round(sup_best,2),
                    avgLev=round(record['leverage']['avg'],2), bestLev=round(lev_best,2)
                    )
        print(logbook.stream)

    return pop, logbook, best_ind


if __name__ == "__main__":
    pop, stats, best_individual = main()
    pop_fit = np.array([ind.fitness.values for ind in pop])

    pf = problem.pareto_front(ref_points)
    print("IGD:", igd(pop_fit, pf))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    p = np.array([ind.fitness.values for ind in pop])
    ax.scatter(p[:, 0], p[:, 1], marker="o", s=24, label="Final Population")

    ax.scatter(pf[:, 0], pf[:, 1], marker="x", c="k", s=32, label="Ideal Pareto Front")

    ref_points = tools.uniform_reference_points(NOBJ, 20)
    ax.scatter(ref_points[:, 0], ref_points[:, 1], marker="o", s=24, label="Reference Points")

    # Añadir el mejor individuo al gráfico
    best_fit = best_individual.fitness.values
    ax.scatter(best_fit[0], best_fit[1], marker="*", c="r", s=100, label="Best Individual")
    
    # Etiquetar las coordenadas del mejor individuo
    ax.annotate(f"({best_fit[0]:.2f}, {best_fit[1]:.2f})",
                xy=(best_fit[0], best_fit[1]), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # Nombres de los ejes
    ax.set_xlabel("Soporte")
    ax.set_ylabel("Leverage")

    ax.legend()
    plt.tight_layout()
    plt.savefig("nsga3_g1.png")
    plt.show()