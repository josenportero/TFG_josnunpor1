from chromosome_simple import Chromosome
from operators_simple import Operators
from metrics_simple import Metrics
from dataset import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
from scoop import futures

def log_results(data, pop, logbook, hof, file_path='C:/Users/Jose/Desktop/TFG/out/results_fitness.txt'):
    # Precision, apalancamiento, kulzcynski
    with open(file_path, 'a') as f:
        f.write("Estadísticas de cada generación:\n")
        for record in logbook:
            f.write(f"Gen: {record['gen']} - Avg: {record['avg']} - Std: {record['std']} - Min: {record['min']} - Max: {record['max']}\n")
        f.write("\nMejores individuos en el Hall of Fame:\n")
        for ind in hof:
            f.write(f"{ind} - Fitness: {np.round(ind.fitness.values, 2)}\n")
        f.write("\n")

        f.write("Métricas de la última generación:\n")
        f.write("Individuo;Soporte;Confianza;Lift;Ganancia;Convicción;Porcentaje de instancias cubiertas;Factor de certeza normalizado\n")
        i = 0
        for ind in pop:
            i += 1
            support = ind.support[2]
            confidence = ind.support[2] / ind.support[0] if ind.support[0] != 0 else 0.
            lift = ind.support[2] / (ind.support[0] * ind.support[1]) if (ind.support[0] != 0) & (ind.support[1] != 0) else 0.
            gain = confidence - ind.support[1]  # Calcular ganancia
            conviction = (1 - ind.support[1]) / (1 - confidence) if confidence != 1. else float('inf')  # Calcular convicción
            coverage_percentage = Metrics.measure_recovered(data, [ind])  # Calcular porcentaje de instancias del dataset que cubre
            normalized_certainty_factor = Metrics.calculate_certainty_factor(data, ind.intervals, ind.transactions)  # Calcular factor de certeza normalizado
            f.write(f"{i};{support};{confidence};{lift};{gain};{conviction};{coverage_percentage};{normalized_certainty_factor}\n")

        recov = Metrics.measure_recovered(data, pop)
        f.write(f"El porcentaje total de instancias cubierto por las reglas es {recov * 100}% \n")

def get_fitness_values(ind):
    return ind.fitness.values if ind.fitness.values is not None else 0.

def get_support(ind):
    return ind.support[2]

def get_confidence(ind):
    return ind.support[2] / ind.support[0] if ind.support[0] != 0 else 0.

def get_lift(ind):
    return ind.support[2] / (ind.support[0] * ind.support[1]) if (ind.support[0] != 0) & (ind.support[1] != 0) else 0.

def main():
    #### Datos para la comprobación del funcionamiento de las clases
    # Toolbox para configurar los algoritmos genéticos
    toolbox = base.Toolbox()

    data = pd.read_csv("C:/Users/Jose/Desktop/TFG/data/FA(in).csv", header=0, sep=';')
    
    df = pd.DataFrame(data)

    w = [0.5, 0.5, 0.0, 0.0, 0.0]

    toolbox.register("dataset", Dataset, dataframe=df)
    print(toolbox.dataset().column_ranges)

    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", Chromosome, fitness=creator.Fitness)

    toolbox.register("individual", Chromosome.create_chromosome, dataset=toolbox.dataset())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", Chromosome.chromosome_eval, dataset=toolbox.dataset())
    toolbox.register("mate", Operators.crossover, dataset=toolbox.dataset())
    toolbox.register("mutate", Operators.mutation, dataset=toolbox.dataset())
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("map", futures.map)

    ngen = 100  # Número de generaciones
    npop = 50  # Número de individuos en población
    tol = 0.001  # Umbral de mejora mínima por generación
    convergence_generations = 10  # Número de generaciones en los que buscar convergencia
    pop = toolbox.population(n=npop)
    hof_size = 6

    hof = tools.HallOfFame(hof_size)

    stats_fit = tools.Statistics(get_fitness_values)
    stats_sup = tools.Statistics(get_support)
    stats_conf = tools.Statistics(get_confidence)
    stats_lift = tools.Statistics(get_lift)

    # Estadísticas de fitness
    stats_fit.register("avg", np.mean, axis=0)
    stats_fit.register("std", np.std, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)
    
    # Estadísticas de soporte
    stats_sup.register("avgSupport", np.mean, axis=0)
    # stats_sup.register("maxSupport", np.max, axis=0)

    # Estadísticas de confianza
    stats_conf.register("avgConfidence", np.mean, axis=0)
    stats_conf.register("maxConfidence", np.max, axis=0)

    # Estadísticas de lift
    stats_lift.register("avgLift", np.mean, axis=0)
    stats_lift.register("maxLift", np.max, axis=0)

    mstats = tools.MultiStatistics(fitness=stats_fit, support=stats_sup, confidence=stats_conf, lift=stats_lift)
   
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max", "avgSupport", "maxSupport", "avgConfidence", "maxConfidence", "avgLift", "maxLift"
    
    # Evaluar individuos con fitness invalido - inicialmente ninguno
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Inicialmente, hof deberá estar vacío    
    hof.update(pop)

    best_ind = tools.selBest(pop, 1)[0]
    sup_best = best_ind.support[2]
    conf_best = best_ind.support[2] / best_ind.support[0] if best_ind.support[0] != 0 else 0.
    lift_best = best_ind.support[2] / (best_ind.support[0] * best_ind.support[1]) if (best_ind.support[0] != 0.) & (best_ind.support[1] != 0.) else 0.
    record = mstats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), avg=round(record['fitness']['avg'][0], 2), std=round(record['fitness']['std'][0], 2), min=round(record['fitness']['min'][0], 2),
                   max=round(record['fitness']['max'][0], 2),
                   avgSupport=round(record['support']['avgSupport'], 2), maxSupport=round(sup_best, 2),
                   avgConfidence=round(record['confidence']['avgConfidence'], 2), maxConfidence=round(conf_best, 2),
                   avgLift=round(record['lift']['avgLift'], 2), maxLift=round(lift_best, 2))
    print(logbook.stream)
    
    ls = []
    fitness_history = []
    support_history = []
    conf_hist = []
    avg_fitness_history = []
    avg_support_history = []
    avg_conf_history = []
    for gen in range(1, ngen + 1):
        ### eaSimple hecho de manera manual para dibujar gráficas
        # Seleccionar individuos de la siguiente poblacion
        offspring = toolbox.select(pop, len(pop) - hof_size)

        # Mutación y reproducción de individuos
        offspring = algorithms.varAnd(offspring, toolbox, 0.8, 0.1)

        # Evaluación de individuos con fitness no válido
        # Los individuos con fitness no válido son los hijos.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            #print(fit)
            ind.fitness.values = fit

        # Extendemos la población con los individuos del hof
        offspring.extend(hof.items)

        # Actualización del hall of fame con los mejores individuos
        hof.update(offspring)

        # Reemplazamos la antigua población por la nueva
        pop[:] = offspring


        # Selección del mejor individuo de cada generación
        best_ind = tools.selBest(pop, 1)[0]
        #print(best_ind)
        sup_best = best_ind.support[2]
        conf_best = best_ind.support[2]/best_ind.support[0] if best_ind.support[0]!=0 else 0.
        lift_best = best_ind.support[2]/(best_ind.support[0]*best_ind.support[1]) if (best_ind.support[0]!=0.) & (best_ind.support[1]!=0.) else 0.
        ls.append(best_ind)

        # Incorporar las estadísticas de la población al log
        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), avg=round(record['fitness']['avg'][0],2), std=round(record['fitness']['std'][0], 2), min=round(record['fitness']['min'][0],2),
                    max=round(record['fitness']['max'][0],2),
                   avgSupport=round(record['support']['avgSupport'],2), maxSupport=round(sup_best,2),
                   avgConfidence=round(record['confidence']['avgConfidence'],2), maxConfidence=round(conf_best,2),
                   avgLift=round(record['lift']['avgLift'],2), maxLift=round(lift_best, 2))
        print(logbook.stream)


        # Fitness del mejor individuo de cada generación
        fitness_history.append(best_ind.fitness.values[0])
        support_history.append(best_ind.support[2])
        conf_hist.append(best_ind.support[2]/best_ind.support[0] if best_ind.support[0]!=0 else 0.)
        avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in pop]))
        avg_support_history.append(np.mean([ind.support[2] for ind in pop]))
        avg_conf_history.append(np.mean([ind.support[2]/ind.support[0] if ind.support[0]!=0 else 0. for ind in pop ]))

        # Convergencia cuando no se produzca mejora razonable en fitness de mejor individuo de 
        # la poblacion en un número determinado de generaciones (convergence_generations)
        if gen >= convergence_generations:
            recent_fitness = fitness_history[-convergence_generations:]
            recent_avg_fitness = avg_fitness_history[-convergence_generations:]
            if max(recent_fitness) - min(recent_fitness) < tol:
                print(f"Parada por convergencia de mejor individuo en generacion: {gen}")
                break
            if max(recent_avg_fitness) - min(recent_avg_fitness) < tol:
                print(f"Parada por convergencia de fitness medio en generacion: {gen}")
                break

    print('Mejores individuos en el Hall Of Fame: \n')
    for ind in hof:
        print(ind, " con función de fitness: ", np.round(ind.fitness.values,2))

    log_results(toolbox.dataset(), pop, logbook, hof)

    # Para dibujar las gráficas de fitness
    fitness_vals = []
    for _, ind in enumerate(ls, start=1):
        fitness_vals.append(ind.fitness.values[0])

    # Plots
    num_generations = len(fitness_vals)  # Ajuste del número de generaciones consideradas
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_generations + 1), fitness_vals, marker='o', linestyle='-', color='b', label='Best Fitness')
    plt.plot(range(1, num_generations + 1), avg_fitness_history[:num_generations], marker='x', linestyle='--', color='r', label='Average Fitness')
    plt.title('Fitness del Mejor Individuo y Promedio por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('Fitness')
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

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_generations + 1), conf_hist, marker='o', linestyle='-', color='g', label='Max. Support')
    plt.plot(range(1, num_generations + 1), avg_conf_history[:num_generations], marker='x', linestyle='--', color='k', label='Average Support')
    plt.title('Confianza del Mejor Individuo y Promedio por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('Confianza')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
