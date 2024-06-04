import random
import pandas as pd
from metrics import Metrics
from dataset import Dataset
from deap import base
from deap import creator
from deap import tools


#### CONSTANTES DEFINIDAS
MAX_PER_TYPE = [2,2] # Lista con el número de atributos máximo que queremos en antecedente y consecuente

class Chromosome:
    '''
    La clase cromosoma constituira la codificacion de las reglas de asociacion.
    Para tal fin, dotaremos a cada cromosoma de los siguientes atributos:
        - Intervals: lista de valores reales que, dos a dos, representan los extremos
        inferior y superior de cada atributo en una regla de asociacion.
        - transactions: lista con tantas posiciones como atributos y que toma los valores
        0, si el atributo correspondiente no aparece en la regla de asociacion, 1 si aparece
        en el antecedente y 2 si aparece en el consecuente.
        - counter_transaction_types: en base a la lista de transacciones, devuelve una lista
        de tamaño dos, de tal manera que el primer elemento de la lista representa el numero
        de atributos que en la regla de asociacion aparecen en el antecedente y el segundo,
        el numero de atributos que aparecen en el consecuente.
        - support: almacena el soporte calculado para el antecedente, consecuente y para
        la propia regla de asociacion en una lista con tres posiciones, a fin de evitar 
        operaciones redundantes al evaluar el fitness de cada cromosoma.
        - fitness: instancia de la clase base.Fitness de DEAP, usada para calcular la
        'medida de bondad' de cada regla de asociacion concreta, en funcion del criterio que
        se haya establecido.
    '''

    def __init__(self, intervals=None, transactions=None, dataset=None, w=[1.0,1.0,-1.0,1.0,-1.0], NOBJ=5):

        self.intervals = intervals if intervals else []
        self.transactions = transactions if transactions else []
        
        self.counter_transaction_type = self.count_transactions() 
        self.support = self.calculate_support(dataset) 
        self.fitness = creator.FitnessMulti()
        self.fitness.values= (Metrics.fitness(self, dataset, w)) if dataset is not None else (0.,)*NOBJ


    def create_chromosome(dataset):
        intervalos = []
        transacciones = []
        count = [0,0]
        for c in dataset.column_ranges:
            min_val = dataset.column_ranges[c]['min']
            max_val = dataset.column_ranges[c]['max']
            inf = random.uniform(min_val, max_val)
            sup = random.uniform(inf, max_val)
            intervalos.extend([inf, sup])
            t = random.choice(dataset.column_ranges[c]['possible transactions'])
            if t == 1:
                count[0] += 1
                res = t if count[0] <= MAX_PER_TYPE[0] else 0
                transacciones.append(res)
            elif t == 2:
                count[1] += 1
                res = t if count[1] <= MAX_PER_TYPE[1] else 0
                transacciones.append(res)
            else:
                transacciones.append(t)

        return Chromosome(intervalos, transacciones, dataset)

    def count_transactions(self):
        contador = [0, 0]  # Inicializamos un contador para transacciones en antecedente y consecuente (resto no aparecen en regla)
        for t in self.transactions:
            if t not in [0, 1, 2]:
                raise ValueError("Las transacciones solo pueden ser 0, 1 o 2")
            if t in [1,2]:
                contador[t-1]+=1
        return contador
    
    def calculate_support(self, dataset):
        '''
        Mediante la definición de esta función, guardamos el soporte de una regla de asociación (cromosoma)
        como un atributo propio de la clase Cromosoma, para evitar recalcular constantemente este valor.
        '''
        intervals = self.intervals
        transactions = self.transactions
        return Metrics.calculate_support(dataset, intervals, transactions) if dataset is not None else []


    # Función de evaluación
    def chromosome_eval(self, dataset, w):
        return Metrics.fitness(self, dataset, w) if dataset is not None else (0.,)
    
    
'''
# Definiciones creator DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
#creator.create("ChromosomeDEAP", Chromosome, fitness=creator.FitnessMax)

# Toolbox para configurar los algoritmos genéticos
toolbox = base.Toolbox()

data = {
    'A': [1.2, 2.3, 5.6],
    'B': [7.8, 8.9, 3.3],
    'C': [9.1, 3.2, 4.8]
}

df = pd.DataFrame(data)

toolbox.register("dataset", Dataset, dataset=df)

# Añadimos al toolbox el cromosoma y la poblacion formada por cromosomas
toolbox.register("chromosome", Chromosome.create_chromosome, toolbox.dataset())
toolbox.register("population", tools.initRepeat, list, toolbox.chromosome, 20) # posteriormente para crear una poblacion
print(toolbox.chromosome().intervals)

# Ejemplo de uso
####### DATOS 'DE JUGUETE' PARA COMPROBACIÓN DE LAS MÉTRICAS
#data = {
#    'A': [1.2, 2.3, 3.4, 4.5, 5.6],
#    'B': [7.8, 8.9, 9.0, 10.1, 11.2],
#    'C': [13.4, 14.5, 15.6, 16.7, 17.8],
#    'D': [19.0, 20.1, 21.2, 22.3, 23.4],
#    'E': [25.6, 26.7, 27.8, 28.9, 30.0]
#}

#df = pd.DataFrame(data)


toolbox.register("evaluate", Chromosome.chromosome_eval, dataset=df)

#print(toolbox.dataset().dataframe)
#cromosoma_ejemplo = toolbox.chromosome()
#print(cromosoma_ejemplo.intervals)
#aptitud_ejemplo = toolbox.evaluate(cromosoma_ejemplo)
#print("#### CROMOSOMA EJEMPLO ####")
#print("Primer nivel del cromosoma de ejemplo: ", cromosoma_ejemplo.intervals)
#print("Segundo nivel del cromosoma de ejemplo:", cromosoma_ejemplo.transactions)
#print("Contador por cada tipo de transacción: ", cromosoma_ejemplo.counter_transaction_type)
#print("Aptitud del cromosoma:", aptitud_ejemplo)

#poblacion = toolbox.population()
#print(poblacion)
'''
