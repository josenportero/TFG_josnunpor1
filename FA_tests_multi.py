from chromosome import Chromosome
from operators import Operators
from dataset import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

def main():
    #### Datos para la comprobación del funcionamiento de las clases

    # DEAP Initialization

    # Toolbox para configurar los algoritmos genéticos
    toolbox = base.Toolbox()

    data = pd.read_csv("C:/Users/Jose/Desktop/TFG/data/FA(in).csv", header=0, sep=';')
    df = pd.DataFrame(data)

    w=[0.2,0.2,0.2,0.2,0.2]

    toolbox.register("dataset", Dataset, dataset=df)

    # Definir tipos de Fitness y Individuo para multiobjetivo
    creator.create("FitnessMulti", base.Fitness, weights=(1.,1., -1., 1., -1.))
    creator.create("Individual", Chromosome, fitness=creator.FitnessMulti)

    toolbox.register("individual", Chromosome.create_chromosome, dataset=toolbox.dataset())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: ind.chromosome_eval(toolbox.dataset(), w))
    toolbox.register("mate", Operators.crossover, dataset=toolbox.dataset())
    toolbox.register("mutate", Operators.mutation, dataset=toolbox.dataset())

    # Definir puntos de referencia para NSGA-III
    ref_points = tools.uniform_reference_points(nobj=5, p=20)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    ngen = 100  # Generations
    npop = 20   # Population
    tol = 0.001  # Convergence threshold for fitness improvement
    convergence_generations = 10   # Number of generations to check for convergence

    pop = toolbox.population(n=npop)

    hof = tools.ParetoFront()
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_sup = tools.Statistics(lambda ind : ind.support[2])

    # Estadísticas fitness
    stats_fit.register("avg", lambda x : np.round(np.mean(x, axis= 0), 3))
    stats_fit.register("std", lambda x : np.round(np.std(x, axis= 0), 3))
    stats_fit.register("min", lambda x : np.round(np.min(x, axis= 0), 3))
    stats_fit.register("max", lambda x : np.round(np.max(x, axis= 0), 3))

    # Estadísticas de soporte
    stats_sup.register("avgSupport", lambda x : np.round(np.mean(x, axis=0),3))
    stats_sup.register("maxSupport", lambda x : np.round(np.max(x, axis=0),3))

    mstats = tools.MultiStatistics(fitness=stats_fit, support=stats_sup)


    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max", "avgSupport", "maxSupport"
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Inicialmente, hof deberá estar vacío
    hof.update(pop)

    record = mstats.compile(pop)
    logbook.record(gen=gen, nevals=len(invalid_ind), avg=record['fitness']['avg'][0], std=record['fitness']['std'][0], min=record['fitness']['min'][0],
                    max=record['fitness']['max'][0],
                   avgSupport=record['support']['avgSupport'], maxSupport=record['support']['maxSupport'])
    print(logbook.stream)

    ls = []
    fitness_history = []
    avg_fitness_history = []
    support_history = []
    avg_support_history = []
    for gen in range(1, ngen + 1):
        # Seleccionar individuos de la siguiente población
        offspring = toolbox.select(pop, len(pop))

        # Operaciones de cruce y mutación sobre los elegidos
        offspring = algorithms.varAnd(offspring, toolbox, 0.8, 0.1)

        # Evaluación de individuos con fitness no válido
        # Los individuos con fitness no válido son los hijos.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Actualizamos el frente de Pareto con los mejores individuos
        hof.update(offspring)

        # Cambiamos la población por la nueva
        pop[:] = offspring

        # Incorporar estadísticas de la población actual al log
        record = mstats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), avg=record['fitness']['avg'][0], std=record['fitness']['std'][0], min=record['fitness']['min'][0],
                    max=record['fitness']['max'][0],
                   avgSupport=record['support']['avgSupport'], maxSupport=record['support']['maxSupport'])
        print(logbook.stream)

        # Individuo con mejor fitness se selecciona y añade en una lista
        best_ind = tools.selBest(pop, 1)[0]
        ls.append(best_ind)

        # Guardamos el valor del fitness del mejor individuo de la generacion
        fitness_history.append(best_ind.fitness.values)
        avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in pop]))
        support_history.append(best_ind.support)
        avg_support_history.append(np.mean([ind.support for ind in pop]))

        # Comprobar convergencia utilizando el mejor individuo
        if gen >= convergence_generations:
            recent_fitness = fitness_history[-convergence_generations:]
            recent_fitness_values = np.array(recent_fitness)
            if np.all(np.max(recent_fitness_values, axis=0) - np.min(recent_fitness_values, axis=0) < tol):
                print(f"Stopping early due to convergence at generation {gen}")
                break

    print('Mejores individuos en el Pareto Front: \n')
    for ind in hof:
        print(ind, " con función de fitness: ", np.round(ind.fitness.values, 2))
        # print(ind.transactions)
        # print(ind.intervals)

    fitness_vals = []
    for _, ind in enumerate(ls, start=1):
        fitness_vals.append(ind.fitness.values)

    # Gráficas
    num_generations = len(fitness_vals)  # Adjust number of generations to plot
    fitness_array = np.array(fitness_vals)
    plt.figure(figsize=(10, 5))
    for i in range(fitness_array.shape[1]):
        plt.plot(range(1, num_generations + 1), fitness_array[:, i], marker='o', linestyle='-', label=f'Objective {i+1}')
    plt.title('Fitness del Mejor Individuo por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('Fitness del Mejor Individuo')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_generations + 1), support_history, marker='o', linestyle='-', color='g', label='Max. Support')
    plt.plot(range(1, num_generations + 1), avg_support_history[:num_generations], marker='x', linestyle='--', color='k', label='Average Support')
    plt.title('Soporte del Mejor Individuo y Promedio por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('Soporte')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()