from chromosome_simple import Chromosome
from metrics_simple import Metrics
from operators_simple import Operators
from dataset import Dataset
import pandas as pd
from deap import base, creator, tools



class Tests:

    def chromosome_test(toolbox):
        cromosoma_ejemplo = toolbox.individual()
        #aptitud_ejemplo = toolbox.evaluate(cromosoma_ejemplo)
        '''
        print("#### CROMOSOMA EJEMPLO ####")
        print("Primer nivel del cromosoma de ejemplo: ", cromosoma_ejemplo.intervals)
        print("Segundo nivel del cromosoma de ejemplo:", cromosoma_ejemplo.transactions)
        print("Contador por cada tipo de transacción: ", cromosoma_ejemplo.counter_transaction_type)
        print("Aptitud del cromosoma:", aptitud_ejemplo)

        #poblacion = toolbox.population()
        #print(poblacion)
        '''
        return cromosoma_ejemplo
    
    def dataset_test(toolbox):
        print("Dataframe: ", toolbox.dataset().dataframe)
        print("Diccionario de constraints por atributo: ", toolbox.dataset().column_ranges)

    def operators_test(toolbox, ind1, ind2, dataset):
        print("Antes del cruce:")
        print("Individuo 1 - Intervalos:", ind1.intervals)
        print("Individuo 1 - Transacciones:", ind1.transactions)
        print("Individuo 2 - Intervalos:", ind2.intervals)
        print("Individuo 2 - Transacciones:", ind2.transactions)

        ind_cruce = toolbox.mate(ind1, ind2,dataset)

        print("\nDespués del cruce:")
        print("Individuo Cruce 1 - Intervalos:", ind_cruce[0].intervals)
        print("Individuo Cruce 1 - Transacciones:", ind_cruce[0].transactions)
        print("Individuo Cruce 2 - Intervalos:", ind_cruce[1].intervals)
        print("Individuo Cruce 2 - Transacciones:", ind_cruce[1].transactions)
        
        print("\nDespués de la mutación:")
        indm = toolbox.mutate(ind1)
        print("Individuo 1 - Intervalos:", indm[0].intervals)
        print("Individuo 1 - Transacciones:", indm[0].transactions)

    def metrics_test(toolbox, c, w):
        print("\n Cromosoma ")
        print(c)
        print("Soporte: ", Metrics.calculate_support(toolbox.dataset(), c.intervals, c.transactions))
        print("Confianza de la regla: ", Metrics.calculate_confidence(toolbox.dataset(), c.intervals, c.transactions))
        print("Lift: ", Metrics.calculate_lift(toolbox.dataset(), c.intervals, c.transactions))
        print("Instancias que 'cubre' la regla: ", Metrics.covered_by_rule(toolbox.dataset(), c.intervals, c.transactions))
        print("Recov: ", Metrics.measure_recovered(toolbox.dataset(), [c]))
        print("Factor de certeza normalizado: ", Metrics.calculate_certainty_factor(toolbox.dataset(), c.intervals, c.transactions))
        print('Fitness: ', Metrics.fitness(c, toolbox.dataset(), w))
        print('Resultado fitness a través de toolbox.evaluate: ',toolbox.evaluate(c))


def main():
    #### Datos para la comprobación del funcionamiento de las clases
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Toolbox para configurar los algoritmos genéticos
    toolbox = base.Toolbox()



    data = {
        'A': [1.2, 2.3, 3.4, 4.5, 5.6],
        'B': [7.8, 8.9, 9.0, 10.1, 11.2],
        'C': [13.4, 14.5, 15.6, 16.7, 17.8],
        'D': [19.0, 20.1, 21.2, 22.3, 23.4],
        'E': [25.6, 26.7, 27.8, 28.9, 30.0]
    }

    # data = {
    #     'A': [1.2, 2.3, 5.6],
    #     'B': [7.8, 8.9, 3.3],
    #     'C': [9.1, 3.2, 4.8]
    # }

    df = pd.DataFrame(data)

    toolbox.register("dataset", Dataset, dataset=df)
    #Tests.dataset_test(toolbox)
    
    # Añadimos al toolbox el cromosoma y la poblacion formada por cromosomas
    #toolbox.register("intervals", random.uniform, min_val=0., max_val=100., n=20)
    #toolbox.register("transactions", random.choices, [0, 1, 2], k=10)  # Generar transacciones aleatorias
    toolbox.register("individual", Chromosome.create_chromosome, dataset=toolbox.dataset())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, 20) # posteriormente para crear una poblacion

    w=[1.,1.,1.,1.,1.]
    toolbox.register("mate", Operators.crossover)
    toolbox.register("mutate", Operators.mutation, dataset=toolbox.dataset())
    toolbox.register("evaluate", Chromosome.chromosome_eval, dataset=toolbox.dataset(), w=w)  

    ind1=Tests.chromosome_test(toolbox)
    ind2=Tests.chromosome_test(toolbox)

    Tests.operators_test(toolbox, ind1, ind2, toolbox.dataset())
    #Tests.metrics_test(toolbox, ind1, w)
    #Tests.metrics_test(toolbox, ind2, w)


if __name__ == "__main__":
    main()


'''
##### ANTES:
# Definir toolbox y registrando los operadores genéticos
toolbox = base.Toolbox()
toolbox.register("mate", Operators.crossover)
toolbox.register("mutate", Operators.mutation)

# Ejemplo de uso de los operadores
ind1 = Chromosome.create_chromosome(10, 0., 100.)
ind2 = Chromosome.create_chromosome(10, 0., 100.)
ls=ind1.intervals
print("Antes del cruce:")
print("Individuo 1 - Intervalos:", ind1.intervals)
print("Individuo 1 - Transacciones:", ind1.transactions)
#print("Individuo 2 - Intervalos:", ind2.intervals)
#print("Individuo 2 - Transacciones:", ind2.transactions)


ind_cruce = toolbox.mate(ind1, ind2)

print("\nDespués del cruce:")
print("Individuo Cruce - Intervalos:", ind_cruce.intervals)
print("Individuo Cruce - Transacciones:", ind_cruce.transactions)


print("\nDespués de la mutación:")
ind1 = toolbox.mutate(ind1)
#ind2 = toolbox.mutate(ind2)
print("Individuo 1 - Intervalos:", ind1.intervals)
print("Individuo 1 - Transacciones:", ind1.transactions)

print("Individuo 2 - Intervalos:", ind2.intervals)
print("Individuo 2 - Transacciones:", ind2.transactions)
'''
'''
#### PARA LA COMPROBACIÓN DE METRICS
# Creación de población de cromosomas para comprobar las métricas
population = [Chromosome.create_chromosome(5, 0, 30) for _ in range(10)]  # Creamos 1 cromosomas de ejemplo

w=[1.,1.,1.,1.,1.]
i=0
for p in population:
    i+=1
    print("\n Cromosoma n = ", i)
    print("Intervalos: ", p.intervals)
    print("Transacciones", p.transactions)
    print("Soporte: ", Metrics.calculate_support(df, p.intervals, p.transactions))
    print("Confianza de la regla: ", Metrics.calculate_confidence(df, p.intervals, p.transactions))
    print("Lift: ", Metrics.calculate_lift(df, p.intervals, p.transactions))
    print("Instancias que 'cubre' la regla: ", Metrics.covered_by_rule(df, p.intervals, p.transactions))
    print("Factor de certeza normalizado: ", Metrics.calculate_certainty_factor(df, p.intervals, p.transactions))
    print('Fitness: ', Metrics.fitness(df, p, w))
print('\n======================================')
print("'Medida recuperada' por las reglas: ", Metrics.measure_recovered(df, population))
'''