import matplotlib.pyplot as plt
import numpy as np
import time
from deap import algorithms, base, creator, tools
import pandas as pd 
from scoop import futures
from chromosome import Chromosome
from operators import Operators
from metrics import Metrics
from dataset import Dataset
from itertools import combinations



NOBJ = 4  # Número de objetivos
MU = 500  # Tamaño de la población
NGEN = 50  # Número de generaciones
CXPB = 0.8  # Probabilidad de cruce
MUTPB = 0.1  # Probabilidad de mutación
HOF_SIZE = int(np.trunc(MU * 0.1)) # tamaño de HOF definido como 10% del tamaño de la poblacion
MAXEXE = 10 # número máximo de ejecuciones del algoritmo (IRL)

# Crear puntos de referencia uniformes para NSGAIII
ref_points = tools.uniform_reference_points(NOBJ, 20)

# Inicialización DEAP
toolbox = base.Toolbox()

toolbox.register("dataset", Dataset)

creator.create("Fitness", base.Fitness, weights=(1., 1., 1.,-1.))
creator.create("Individual", Chromosome, fitness=creator.Fitness)

toolbox.register("individual", Chromosome.create_chromosome)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

Chromosome.MULTI = True
toolbox.register("evaluate", Chromosome.chromosome_eval_multi)
toolbox.register("mate", Operators.crossover)
toolbox.register("mutate", Operators.mutation)

pop = toolbox.population(n=MU)
initial_pop = np.array([ind.fitness.values for ind in pop])

toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
toolbox.register("map", futures.map)

def log_results(pop, logbook, hof, inicio, fin, file_path='C:/Users/Jose/Desktop/TFG/out/pruebas_mo_3.txt'):
    """"
    Guarda estadísticas sobre la ejecución en un fichero a modo de log
    """
    with open(file_path, 'a') as f:
        f.write('\n##################################################\n')
        f.write('\n PRUEBAS PARA TEST MULTIOBJETIVO \n')
        f.write('\n##################################################\n')
        f.write(f'Objetivos probados: (soporte, confianza, cf, ampl) \n')
        f.write(f'Pruebas con {MU} individuos y {NGEN} generaciones \n')
        f.write(f'Tiempo total de ejecución del algoritmo {fin - inicio} segundos \n')
        f.write("Estadísticas de cada generación:\n")
        # for record in logbook:
        #     f.write(f"Gen: {record['gen']} - Avg: {record['avg']} - Std: {record['std']} - Min: {record['min']} - Max: {record['max']}\n")

        f.write("Métricas del Hall Of Fame:\n")
        f.write("ID;Soporte;Confianza;Lift;Leverage;Ganancia;Conviccion;Recov;Chi-sq;CF;Fitness\n")
        j = 0
        i = 0
        n = Dataset.dataframe.shape[0]

        f.write(f'ID; INDIVIDUO\n')
        for ind in hof:
            f.write(f'{i}; {ind} \n')
            i += 1

        for ind in hof:
            support = round(ind.support[2], 3)
            confidence = round(ind.support[2] / ind.support[0], 3) if ind.support[0] != 0 else 0.
            lift = round(ind.support[2] / (ind.support[0] * ind.support[1]), 3) if (ind.support[0] != 0) & (ind.support[1] != 0) else 0.
            gain = round(confidence - ind.support[1], 3)
            conviction = round((1 - ind.support[1]) / (1 - confidence), 3) if confidence != 1. else float('inf')
            lev = round(support - ind.support[1] * ind.support[0], 3)
            coverage = round(sum(Metrics.measure_recovered([ind])), 3) / n
            chisq = Metrics.calculate_chi_squared(ind)
            normalized_certainty_factor = round(Metrics.calculate_certainty_factor(ind), 3)
            fitness = np.round(ind.fitness.values, 2)
            f.write(f"{j};{support};{confidence};{lift};{lev};{gain};{conviction};{coverage};{chisq};{normalized_certainty_factor};{fitness}\n")
            j += 1

        recovp = sum(Metrics.measure_recovered(pop))
        recov = sum(Metrics.measure_recovered(hof))
        f.write(f"Número de veces que se repiten instancias en hof: {recov} \n")
        f.write(f"Número de veces que se repiten instancias en población: {recovp} \n")

def main(seed=None):
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_sup = tools.Statistics(lambda ind: ind.support[2])
    stats_conf = tools.Statistics(lambda ind: Metrics.calculate_confidence(ind))
    stats_cf = tools.Statistics(lambda ind: Metrics.calculate_certainty_factor(ind))

    stats_fit.register("avg", np.mean, axis=0)
    stats_fit.register("std", np.std, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)

    stats_sup.register("avg", np.mean)
    stats_sup.register("max", np.max)

    stats_cf.register("avg", np.mean)
    stats_cf.register("max", np.max)

    stats_conf.register("avg", np.mean)
    stats_conf.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, support=stats_sup, confidence=stats_conf, cf=stats_cf)
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max", "avgSup", "bestSup", "avgConf", "bestConf","avgCF", "bestCF"

    num_exe=0
    irl_individuals = []
    while num_exe < MAXEXE:
        # Inicializacion de la poblacion y el hof
        
        
        hof_aprox = tools.ParetoFront()
        pop = toolbox.population(n=MU)


        # Asignacion de fitness a los cromosomas
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Actualizamos hof y guardamos los mejores individuos
        hof_aprox.update(pop)
        best_inds = tools.selNSGA3(pop, HOF_SIZE, ref_points)


        ### Para guardar y mostrar estadisticas de la evolucion
        sup_evol = []
        conf_evol = []
        cf_evol = []
        pop_evol=[]

        best_ind = hof_aprox[0]
        sup_best = best_ind.support[2]
        conf_best = Metrics.calculate_confidence(best_ind)
        cf_best = Metrics.calculate_certainty_factor(best_ind)
        #lev_best = Metrics.calculate_leverage(best_ind)

        record = mstats.compile(pop)
        sup_evol.append(round(record['support']['avg'], 2))
        conf_evol.append(round(record['confidence']['avg'], 2))
        cf_evol.append(round(record['cf']['avg'], 2))
        logbook.record(gen=0, nevals=len(invalid_ind), avg=np.round((record['fitness']['avg']), 2), std=np.round((record['fitness']['std']), 2),
                    min=np.round((record['fitness']['min']), 2),
                    max=np.round((record['fitness']['max']), 2),
                    avgSup=round(record['support']['avg'], 2), bestSup=round(sup_best, 2),
                    avgConf=round(record['confidence']['avg'], 2), bestConf=round(conf_best, 2),
                    avgCF=round(record['cf']['avg'], 2), bestCF=round(cf_best, 2)
                    )

        print(logbook.stream)

        # Bucle del proceso evolutivo
        for gen in range(1, NGEN):
            offspring = toolbox.select(pop, MU - (HOF_SIZE))
            offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring.extend(best_inds)

            hof_aprox.update(offspring)

            pop[:] = offspring

            best_inds = tools.selNSGA3(pop, HOF_SIZE, ref_points)
            best_ind = hof_aprox[0]
            sup_best = best_ind.support[2]
            conf_best = Metrics.calculate_confidence(best_ind)
            cf_best = Metrics.calculate_certainty_factor(best_ind)
            #lev_best = Metrics.calculate_gain(best_ind)

            # if (gen%10==0):
            #     pop_evol.extend(pop)
            #     for i in pop_evol:
            #         print(i)
            
            

            record = mstats.compile(pop)
            sup_evol.append(round(record['support']['avg'], 2))
            conf_evol.append(round(record['confidence']['avg'], 2))
            cf_evol.append(round(record['cf']['avg'], 2))
            logbook.record(gen=gen, nevals=len(invalid_ind), avg=np.round((record['fitness']['avg']), 2), std=np.round((record['fitness']['std']), 2),
                        min=np.round((record['fitness']['min']), 2),
                        max=np.round((record['fitness']['max']), 2),
                        avgSup=round(record['support']['avg'], 2), bestSup=round(sup_best, 2),
                        avgConf=round(record['confidence']['avg'], 2), bestConf=round(conf_best, 2),
                        avgCF=round(record['cf']['avg'], 2), bestCF=round(cf_best, 2)
                        )
            print(logbook.stream)
            #print(len(pop))
        irl_individuals.append(tools.selNSGA3(hof_aprox,1,ref_points)[0])
        Metrics.recov = Metrics.measure_recovered(irl_individuals)
        num_exe+=1

    return pop, logbook, hof_aprox, sup_evol, conf_evol, cf_evol, irl_individuals


if __name__ == "__main__":
    inicio = time.time()
    pop, stats, hof_aprox, sup_evol, conf_evol, cf_evol, irl_individuals = main()
    pop_fit = np.array([ind.fitness.values for ind in pop])
    fin = time.time()

    log_results(pop, stats, irl_individuals, inicio, fin)

    
   # Listado de variables
variables = ['Soporte', 'Confianza', 'CF', 'Recov']

# Generar todas las combinaciones posibles de tres variables
combinations_of_three = list(combinations(variables, 3))

# Crear gráficos 3D para cada combinación de tres variables
for combo in combinations_of_three:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x_idx = variables.index(combo[0])
    y_idx = variables.index(combo[1])
    z_idx = variables.index(combo[2])
    
    ax.scatter(pop_fit[:, x_idx], pop_fit[:, y_idx], pop_fit[:, z_idx], marker="o", s=24, label="Poblacion Final")
    ax.scatter(initial_pop[:, x_idx], initial_pop[:, y_idx], initial_pop[:, z_idx], marker="o", s=24, label="Poblacion Inicial", alpha=0.6)
    ax.scatter([ind.fitness.values[x_idx] for ind in hof_aprox], 
               [ind.fitness.values[y_idx] for ind in hof_aprox], 
               [ind.fitness.values[z_idx] for ind in hof_aprox], marker="x", c="r", s=32, label="Frente de Pareto Aproximado")
    
    ax.set_xlabel(combo[0])
    ax.set_ylabel(combo[1])
    ax.set_zlabel(combo[2])
    ax.legend()

    plt.title(f'Visualización 3D de Soluciones de Optimización Multiobjetivo ({combo[0]}, {combo[1]}, {combo[2]})')
    plt.show()

    # Gráfica para la evolución del soporte medio
    fig, ax = plt.subplots()
    ax.set_xlabel('Generación')
    ax.set_ylabel('Soporte Medio')
    ax.plot(range(len(sup_evol)), sup_evol, color='tab:blue', label='Soporte Medio')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Gráfica para la evolución de la confianza media
    fig, ax = plt.subplots()
    ax.set_xlabel('Generación')
    ax.set_ylabel('Confianza Media')
    ax.plot(range(len(conf_evol)), conf_evol, color='tab:red', label='Confianza Media')
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('Generación')
    ax.set_ylabel('CF Medio')
    ax.plot(range(len(cf_evol)), cf_evol, color='tab:green', label='CF Medio')
    ax.legend()
    plt.tight_layout()
    plt.show()

