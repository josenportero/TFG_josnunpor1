import random
import pandas as pd
from metrics_simple import Metrics
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
    def __init__(self, intervals=None, transactions=None, dataset=None):
        self.intervals = intervals if intervals else []
        self.transactions = transactions if transactions else []
        self.dataset = dataset
        self.counter_transaction_type = self.count_transactions() 
        self.support = []
        self.fitness = creator.Fitness()
        self.fitness.values= Metrics.fitness(self, dataset) if dataset is not None else 0.

    def __str__(self):
            '''
            Modificación del método str para imprimir las reglas de asociación de manera:
            IF atrib1[rango] AND atrib2[rango] THEN atrib3[rango] AND atrib4[rango] 
            '''
            ant = [i for i in range(len(self.transactions)) if self.transactions[i] == 1]
            cons = [i for i in range(len(self.transactions)) if self.transactions[i] == 2]
            
            a_names = [self.dataset.dataframe.columns[e] for e in ant]
            c_names = [self.dataset.dataframe.columns[e] for e in cons]

            a_interv = [[self.intervals[2*i], self.intervals[2*i+1]] for i in ant]
            c_interv = [[self.intervals[2*i], self.intervals[2*i+1]] for i in cons]

            acum = "IF "
            for i in range(len(ant)):
                if i > 0:
                    acum += "AND "
                acum += f"{a_names[i]} [{a_interv[i][0]:.2f}, {a_interv[i][1]:.2f}] "
            acum += "THEN "
            for i in range(len(cons)):
                if i > 0:
                    acum += "AND "
                acum += f"{c_names[i]} [{c_interv[i][0]:.2f}, {c_interv[i][1]:.2f}] "

            return acum
    
    def __eq__(self, other):
        '''
        Dos cromosomas se considerarán idénticos si los rangos de los atributos que
        están en las reglas son los mismos y están en el mismo 'lado' de la regla,
        es decir, ambos en antecedente o ambos en consecuente.
        '''
        if not isinstance(other, Chromosome):
            return False
        for i in range(len(self.transactions)):
            if self.transactions[i] != 0 or other.transactions[i] != 0:
                if self.transactions[i] != other.transactions[i]:
                    return False
                if self.intervals[2*i:2*i+2] != other.intervals[2*i:2*i+2]:
                    return False
        return True

    def create_chromosome(dataset):
        intervalos = []
        transacciones = []
        count = [0,0,0]
        for c in dataset.column_ranges:
            min_val = dataset.column_ranges[c]['min']
            max_val = dataset.column_ranges[c]['max']
            inf = random.uniform(min_val, max_val)
            sup = random.uniform(inf, max_val)
            intervalos.extend([inf, sup])
            t = random.choice(dataset.column_ranges[c]['possible transactions'])
            if t == 1 and count[1] < MAX_PER_TYPE[0]:
                count[1] += 1
                transacciones.append(t)
            elif t == 2 and count[2] < MAX_PER_TYPE[1]:
                count[2] += 1
                transacciones.append(t)
            else:
                count[0] += 1
                transacciones.append(0)

        if count[1]==0:
            idx = random.choice([i for i, x in enumerate(transacciones) if x == 0])
            transacciones[idx] = 1
            count[1] += 1

        if count[2]==0:
            idx = random.choice([i for i, x in enumerate(transacciones) if x == 0])
            transacciones[idx] = 2
            count[2] += 1

        return Chromosome(intervalos, transacciones, dataset)
    
    def force_valid(intervalos, transacciones, dataset):
        '''
        Método auxiliar, en caso de que los operadores de mutación y cruce no puedan generar un
        individuo válido pasado el número máximo de iteraciones, forzar la creación de uno
        que sí que lo sea.
        '''
        ### POSIBLE MEJORA: por ejemplo en cromosoma con counts=[10,1,7] pasar alguno de los 
        # 7 atributos en consecuente al antecedente, en lugar de quitarlos de la regla.
        c_aux = Chromosome(intervalos, transacciones, dataset)
        counts = c_aux.count_transactions()
        #print(counts)
        if counts[1]==0:
            idx = random.choice([i for i, x in enumerate(transacciones) if x == 0])
            transacciones[idx] = 1 # No actualizamos counts porque no hace falta
        elif counts[1]>MAX_PER_TYPE[0]:
            while counts[1] > MAX_PER_TYPE[0]:
                idx = random.choice([i for i, x in enumerate(transacciones) if x == 0])
                transacciones[idx] = 0
                counts[1]-=1
                counts[0]+=1

        if counts[2]==0:
            idx = random.choice([i for i, x in enumerate(transacciones) if x == 0])
            transacciones[idx] = 2
        elif counts[2]>MAX_PER_TYPE[1]:
            while counts[2] > MAX_PER_TYPE[1]:
                idx = random.choice([i for i, x in enumerate(transacciones) if x == 0])
                transacciones[idx] = 0
                counts[2]-=1
                counts[0]+=1
        #print(counts)
        return Chromosome(intervalos, transacciones, dataset)

    def count_transactions(self):
        contador = [0, 0, 0]  # Inicializamos un contador para transacciones en antecedente y consecuente (resto no aparecen en regla)
        for t in self.transactions:
            if t not in [0, 1, 2]:
                raise ValueError("Las transacciones solo pueden ser 0, 1 o 2")
            if t in [0, 1,2]:
                contador[t]+=1
        return contador
    
    # def calculate_support(self, dataset):
    #     '''
    #     Mediante la definición de esta función, guardamos el soporte de una regla de asociación (cromosoma)
    #     como un atributo propio de la clase Cromosoma, para evitar recalcular constantemente este valor.
    #     '''
    #     intervals = self.intervals
    #     transactions = self.transactions
    #     return Metrics.calculate_support(dataset, intervals, transactions) if dataset is not None else []


    # Función de evaluación
    def chromosome_eval(self, dataset):
        return Metrics.fitness(self, dataset) if dataset is not None else (0.,)
    
'''    
    def validate(self, transactions):
        """
        Comprobación de que el cromosoma cumple las restricciones que se establecen en el paper de QARGA seccion 2.3 (i.e. 
        representa una regla de asociación con sentido).
        """
        antecedents = [t for t in transactions if t == 1]
        consequents = [t for t in transactions if t == 2]
        if len(antecedents) < 1 or len(antecedents) > MAX_PER_TYPE[0] or len(consequents) < 1 or len(consequents) > MAX_PER_TYPE[1]:
            return False
        return True
    

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