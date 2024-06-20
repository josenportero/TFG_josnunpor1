import time
from chromosome import Chromosome
from operators import Operators
from metrics import Metrics
from dataset import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
from scoop import futures

def log_results(pop,  logbook, hof, npop, inicio, fin, file_path='C:/Users/Jose/Desktop/TFG/out/results_esiosv3.txt'):
    # Precision, apalancamiento , kulzcynski
    with open(file_path, 'a') as f:
        f.write('\n##################################################\n')
        f.write('\n PRUEBAS PARA CLASIFICACION DE CONVICTION \n')
        f.write('\n##################################################\n')
        f.write(f'Objetivos probados: soporte, conf, cf, recov\n')
        f.write(f'Pruebas con {npop} individuos\n')
        f.write(f'Tiempo total de ejecución del algoritmo {fin-inicio} segundos \n')
        f.write(f'Vector de pesos para el fitness: {Metrics.W}\n')
        f.write("Estadísticas de cada generación:\n")
        for record in logbook:
            f.write(f"Gen: {record['gen']} - Avg: {record['avg']} - Std: {record['std']} - Min: {record['min']} - Max: {record['max']}\n")

        f.write("Métricas del Hall Of Fame:\n")
        f.write("ID;Soporte;Confianza;Lift;Leverage;Ganancia;Conviccion;Recov;Chi-sq;CF;Fitness\n")
        j=0
        i=0
        n=Dataset.dataframe.shape[0]

        f.write(f'ID; INDIVIDUO\n')
        for ind in pop:
            f.write(f'{i}; {ind} \n')
            i+=1

        for ind in pop:
            
            support = round(ind.support[2],3)
            confidence = round(ind.support[2] / ind.support[0], 3) if ind.support[0] != 0 else 0.
            lift = round(ind.support[2] / (ind.support[0] * ind.support[1]), 3) if (ind.support[0] != 0) & (ind.support[1] != 0) else 0.
            gain = round(confidence - ind.support[1],3)  # Calcular ganancia
            conviction = round((1-ind.support[1])/(1-confidence),3)  if confidence!=1. else float('inf') # Calcular convicción
            lev = round(support-ind.support[1]*ind.support[0], 3)
            coverage = round(sum(Metrics.measure_recovered([ind])),3)/n
            chisq = Metrics.calculate_chi_squared(ind) 
            normalized_certainty_factor = round(Metrics.calculate_certainty_factor(ind),3)  # Calcular factor de certeza normalizado
            fitness = np.round(ind.fitness.values, 2)
            f.write(f"{j};{support};{confidence};{lift};{lev};{gain};{conviction};{coverage};{chisq};{normalized_certainty_factor};{fitness}\n")
            j+=1

        recovp = sum(Metrics.measure_recovered(pop))
        recov = sum(Metrics.measure_recovered(hof))
        f.write(f"Número de veces que se repiten instancias en hof: {recov} \n")
        f.write(f"Número de veces que se repiten instancias en población: {recovp} \n")


def get_fitness_values(ind):
    return ind.fitness.values if ind.fitness.values is not None else 0.

def main():
    #### Datos para la comprobación del funcionamiento de las clases
    time1 = time.time()
    # Inicialización DEAP
    
    # Toolbox para configurar los algoritmos genéticos
    toolbox = base.Toolbox()

    #ruta_fichero = "C:/Users/Jose/Desktop/TFG/data/datos_TFGb.xlsx"
    # data =   pd.read_excel("C:/Users/Jose/Desktop/TFG/data/datos_TFG.xlsx", header=0)
    # df = data.drop(data.columns[0], axis=1)
    # df = df.astype(float)

    toolbox.register("dataset", Dataset)


    #print(toolbox.dataset().dataframe)
    #print(toolbox.dataset().column_ranges)
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", Chromosome, fitness=creator.Fitness)

    toolbox.register("individual", Chromosome.create_chromosome)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", Chromosome.chromosome_eval)
    toolbox.register("mate", Operators.crossover)
    toolbox.register("mutate", Operators.mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evalución paralela para agilizar calculos
    toolbox.register("map", futures.map)

    Chromosome.MULTI = False
    ngen = 50 # Número de generaciones
    npop = 100 # Número de individuos en población
    tol = 0.01  # Umbral de mejora mínima por generación
    convergence_generations = 10   # Número de generaciones en los que buscar convergencia
    pop   = toolbox.population(n=npop)
    max_hof_size = int(np.trunc(0.1*npop))
    hof_size = 1

    hof   = tools.HallOfFame(hof_size)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_sup = tools.Statistics(lambda ind : ind.support[2])
    stats_conf = tools.Statistics(lambda ind : ind.support[2]/ind.support[0] if ind.support[0]!=0 else 0.)
    stats_lift = tools.Statistics(lambda ind : ind.support[2]/(ind.support[0]*ind.support[1]) if (ind.support[0]!=0.) & (ind.support[1]!=0.) else 0.)
    stats_cf = tools.Statistics(lambda ind : Metrics.calculate_certainty_factor(ind))
    #stats_recov = tools.Statistics(lambda ind : sum([1 for i in range(Dataset.dataframe.shape[0]) if Metrics.recov[i] + Metrics.measure_recovered([ind])[i] > 1])/Dataset.dataframe.shape[0])

    # Estadísticas de fitness
    stats_fit.register("avg", np.mean, axis=0)
    stats_fit.register("std", np.std, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)
    
    # Estadísticas de soporte
    stats_sup.register("avgSupport", np.mean, axis=0)
    #stats_sup.register("maxSupport", np.max, axis=0)

    # Estadísticas de confianza
    stats_conf.register("avgConfidence", np.mean, axis=0)
    stats_conf.register("maxConfidence", np.max, axis=0)

    # Estadísticas de lift
    stats_lift.register("avgLift", np.mean, axis=0)
    stats_lift.register("maxLift", np.max, axis=0)

    # Estadísticas de cf
    stats_cf.register("avgCF", np.mean, axis=0)
    stats_cf.register("maxCF", np.max, axis=0)

    # Estadísticas de recov
    #stats_recov.register("avgRecov", np.mean, axis=0)

    mstats = tools.MultiStatistics(fitness=stats_fit, support=stats_sup, confidence=stats_conf, cf=stats_cf)
   
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max", "avgSupport", "maxSupport", "avgConfidence", "maxConfidence", "avgCF", "maxCF"
    
    # Evaluar individuos con fitness invalido - inicialmente ninguno
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # print(f'tiempo sin vectorizar {Metrics.t}')
    # print(f'tiempo vectorizado {Metrics.t_np}')


    # Inicialmente, hof deberá estar vacío    
    hof.update(pop)

    best_ind = tools.selBest(pop, 1)[0]
    sup_best = best_ind.support[2]
    conf_best = best_ind.support[2]/best_ind.support[0] if best_ind.support[0]!=0 else 0.
    lift_best = best_ind.support[2]/(best_ind.support[0]*best_ind.support[1]) if (best_ind.support[0]!=0.) & (best_ind.support[1]!=0.) else 0.
    cf_best = Metrics.calculate_certainty_factor(best_ind)
    record = mstats.compile(pop)
    #print(record)
    
    logbook.record(gen=0, nevals=len(invalid_ind), avg=round(record['fitness']['avg'][0],2), std=round(record['fitness']['std'][0], 2), min=round(record['fitness']['min'][0],2),
                    max=round(record['fitness']['max'][0],2), 
                   avgSupport=round(record['support']['avgSupport'],2), maxSupport=round(sup_best,2),
                   avgConfidence=round(record['confidence']['avgConfidence'],2), maxConfidence=round(conf_best,2),
                   avgCF=round(record['cf']['avgCF'],2), maxCF=round(cf_best, 2))
    print(logbook.stream)

    #print("Soportes calculados: ", Metrics.SUP_CALC)
    ls = []
    fitness_history = []
    support_history = []
    conf_hist = []
    cf_hist = []
    # recov_hist = []
    avg_fitness_history = []
    avg_support_history = []
    avg_conf_history = []
    avg_cf_history = []
    # avg_recov_hist = []
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

        if(hof_size < max_hof_size):
            best_candidates = tools.selBest(offspring,max_hof_size+1)
            for cand in best_candidates:
                if cand not in hof:
                    hof.insert(cand)
                    hof_size += 1
                    break

        Metrics.recov = Metrics.measure_recovered(hof)
        Metrics.HOF = hof
        #print(Metrics.recov)


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
                    avgCF=round(record['cf']['avgCF'],2), maxCF=round(cf_best, 2))
        print(logbook.stream)
        #print("Soportes calculados: ", Metrics.SUP_CALC)


        # Fitness del mejor individuo de cada generación
        fitness_history.append(best_ind.fitness.values[0])
        support_history.append(best_ind.support[2])
        conf_hist.append(best_ind.support[2]/best_ind.support[0] if best_ind.support[0]!=0 else 0.)
        cf_hist.append(Metrics.calculate_certainty_factor(best_ind))
        #recov_hist.append(sum([1 for i in range(Dataset.dataframe.shape[0]) if Metrics.recov[i] + Metrics.measure_recovered([ind])[i] > 1])/Dataset.dataframe.shape[0])
        avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in pop]))
        avg_support_history.append(np.mean([ind.support[2] for ind in pop]))
        avg_conf_history.append(np.mean([ind.support[2]/ind.support[0] if ind.support[0]!=0 else 0. for ind in pop ]))
        avg_cf_history.append(np.mean([Metrics.calculate_certainty_factor(ind) for ind in pop ]))
        #avg_recov_hist.append(np.mean(sum([1 for i in range(Dataset.dataframe.shape[0]) if Metrics.recov[i] + Metrics.measure_recovered([ind])[i] > 1])/Dataset.dataframe.shape[0]))


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



    time2 = time.time()
    print('Mejores individuos en el Hall Of Fame: \n')
    for ind in hof:
        print(ind, " con función de fitness: ", np.round(ind.fitness.values,2))

    log_results(pop, logbook, hof, npop, time1, time2)

    # Para dibujar las gráficas de fitness
    fitness_vals = []
    for _, ind in enumerate(ls, start=1):
        fitness_vals.append(ind.fitness.values[0])

        
    ls = Operators.calculate_ranges()
    i=0
    for c in Dataset.dataframe.columns:
        print(f'Para la columna {c} el rango es [{ls[i]},{ls[i+1]}]')
        i+=2

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
    plt.plot(range(1, num_generations + 1), support_history, marker='o', linestyle='-', color='g', label='Support of best')
    plt.plot(range(1, num_generations + 1), avg_support_history[:num_generations], marker='x', linestyle='--', color='k', label='Average Support')
    plt.title('Soporte del Mejor Individuo y Promedio por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('Soporte')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_generations + 1), conf_hist, marker='o', linestyle='-', color='g', label='Confidence of best')
    plt.plot(range(1, num_generations + 1), avg_conf_history[:num_generations], marker='x', linestyle='--', color='k', label='Average Confidence')
    plt.title('Confianza del Mejor Individuo y Promedio por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('Confianza')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_generations + 1), cf_hist, marker='o', linestyle='-', color='g', label='CF of best')
    plt.plot(range(1, num_generations + 1), avg_cf_history[:num_generations], marker='x', linestyle='--', color='k', label='Average CF')
    plt.title('CF del Mejor Individuo y Promedio por Generación')
    plt.xlabel('Número de Generación')
    plt.ylabel('CF')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, num_generations + 1), recov_hist, marker='o', linestyle='-', color='g', label='Recov of best')
    # plt.plot(range(1, num_generations + 1), avg_recov_hist[:num_generations], marker='x', linestyle='--', color='k', label='Average Recov')
    # plt.title('Recov del Mejor Individuo y Promedio por Generación')
    # plt.xlabel('Número de Generación')
    # plt.ylabel('Recov')
    # plt.legend()
    # plt.grid(True)
    # plt.show()



    
if __name__ == "__main__":
    main()