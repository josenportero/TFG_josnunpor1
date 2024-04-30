from chromosome import Chromosome
from metrics import Metrics
from operators import Operators
import pandas as pd
from deap import base, creator, tools

#### Datos para la comprobación del funcionamiento de las clases
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("ChromosomeDEAP", Chromosome, fitness=creator.FitnessMax)

# Toolbox para configurar los algoritmos genéticos
toolbox = base.Toolbox()

# Añadimos al toolbox el cromosoma y la poblacion formada por cromosomas
#toolbox.register("intervals", random.uniform, min_val=0., max_val=100., n=20)
#toolbox.register("transactions", random.choices, [0, 1, 2], k=10)  # Generar transacciones aleatorias
toolbox.register("chromosome", Chromosome.create_chromosome, n=10, min_val=0., max_val=100.)
toolbox.register("population", tools.initRepeat, list, toolbox.chromosome, 20) # posteriormente para crear una poblacion


data = {
    'A': [1.2, 2.3, 3.4, 4.5, 5.6],
    'B': [7.8, 8.9, 9.0, 10.1, 11.2],
    'C': [13.4, 14.5, 15.6, 16.7, 17.8],
    'D': [19.0, 20.1, 21.2, 22.3, 23.4],
    'E': [25.6, 26.7, 27.8, 28.9, 30.0]
}

df = pd.DataFrame(data)

toolbox.register("mate", Operators.crossover)
toolbox.register("mutate", Operators.mutation)
toolbox.register("evaluate", Chromosome.chromosome_eval, dataset=df)

cromosoma_ejemplo = toolbox.chromosome()
aptitud_ejemplo = toolbox.evaluate(cromosoma_ejemplo)
print("#### CROMOSOMA EJEMPLO ####")
print("Primer nivel del cromosoma de ejemplo: ", cromosoma_ejemplo.intervals)
print("Segundo nivel del cromosoma de ejemplo:", cromosoma_ejemplo.transactions)
print("Contador por cada tipo de transacción: ", cromosoma_ejemplo.counter_transaction_type)
print("Aptitud del cromosoma:", aptitud_ejemplo)

#poblacion = toolbox.population()
#print(poblacion)


'''
##### PARA COMPROBAR EL FUNCIONAMIENTO DE LOS OPERADORES
# Definir toolbox y registrando los operadores genéticos
toolbox = base.Toolbox()
toolbox.register("mate", Operators.crossover)
toolbox.register("mutate", Operators.mutation)

# Ejemplo de uso de los operadores
ind1 = Chromosome.create_chromosome(10, 0., 100.)
ind2 = Chromosome.create_chromosome(10, 0., 100.)
MIN_MAX_LIST=ind1.intervals
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