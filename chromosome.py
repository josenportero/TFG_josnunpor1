import random
import pandas as pd
from metrics import Metrics
from dataset import Dataset
from deap import base
from deap import creator
from deap import tools
import numpy as np


#### CONSTANTES DEFINIDAS
 # Lista con el número de atributos máximo que queremos en antecedente y consecuente

class Chromosome:
    '''
    La clase cromosoma constituira la codificacion de las reglas de asociacion.
    Para tal fin, dotaremos a cada cromosoma de los siguientes atributos:
        - Intervals: lista de valores reales que, dos a dos, representan los extremos
        inferior y superior de cada atributo en una regla de asociacion.
        - types: lista con tantas posiciones como atributos y que toma los valores
        0, si el atributo correspondiente no aparece en la regla de asociacion, 1 si aparece
        en el antecedente y 2 si aparece en el consecuente.
        - counter_transaction_types: en base a la lista de tipos, devuelve una lista
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
    MULTI = False
    MAX_PER_TYPE = [3,1]
    def __init__(self, intervals=None, types=None):
        self.intervals = intervals if intervals else []
        self.types = types if types else []
        self.counter_types = self.count_types() 
        self.support = []
        self.fitness = creator.Fitness()
        if Chromosome.MULTI == True:
            self.fitness.values= Metrics.fitness_multi(self)
        else:
            self.fitness.values= Metrics.fitness(self) 

    def __str__(self):
            '''
            Modificación del método str para imprimir las reglas de asociación de manera:
            IF atrib1[rango] AND atrib2[rango] THEN atrib3[rango] AND atrib4[rango] 
            '''
            ant = [i for i in range(len(self.types)) if self.types[i] == 1]
            cons = [i for i in range(len(self.types)) if self.types[i] == 2]
            
            a_names = [Dataset.dataframe.columns[e] for e in ant]
            c_names = [Dataset.dataframe.columns[e] for e in cons]

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
        for i in range(len(self.types)):
            if self.types[i] != 0 or other.types[i] != 0:
                if self.types[i] != other.types[i]:
                    return False
                if not np.allclose(self.intervals[2*i:2*i+2], other.intervals[2*i:2*i+2], atol=0.01):
                    return False
        return True

    def create_chromosome():
        intervalos = []
        tipos = [0 for _ in range(len(Dataset.dataframe.columns))]
        count = [0,0,0]
        possible_cons = [i for i,c in enumerate(Dataset.dataframe.columns) if 2 in Dataset.column_ranges[c]['possible types']]
        possible_ants = [i for i,c in enumerate(Dataset.dataframe.columns) if (1 in Dataset.column_ranges[c]['possible types'] and c not in possible_cons)]

        index_ants = random.sample(possible_ants, Chromosome.MAX_PER_TYPE[0])
        index_cons = random.sample(possible_cons, Chromosome.MAX_PER_TYPE[1])

        for i,c in enumerate(Dataset.dataframe.columns):
            min_val = Dataset.column_ranges[c]['min']
            max_val = Dataset.column_ranges[c]['max']
            inf = random.uniform(min_val, max_val)
            sup = random.uniform(inf, max_val)
            intervalos.extend([inf, sup])
            # if c in indexes:
            #     t = random.choice(Dataset.column_ranges[c]['possible types'])
            #     if t == 1 and count[1] < Chromosome.MAX_PER_TYPE[0]:
            #         count[1] += 1
            #         tipos.append(t)
            #     elif t == 2 and count[2] < Chromosome.MAX_PER_TYPE[1]:
            #         count[2] += 1
            #         tipos.append(t)
            #     else:
            #         count[0] += 1
            #         tipos.append(0)

            if i in index_cons:
                count[2] +=1
                tipos[i]=2
            elif i in index_ants:
                count[1]+=1
                tipos[i]=1

        # if count[1]==0:
        #     idx = random.choice([i for i, x in enumerate(tipos) if x == 0])
        #     tipos[idx] = 1
        #     count[1] += 1

        # if count[2]==0:
        #     idx = random.choice([i for i, x in enumerate(tipos) if x == 0])
        #     tipos[idx] = 2
        #     count[2] += 1

        return Chromosome(intervalos, tipos)
    
    def force_valid(intervalos, tipos):
        '''
        Método auxiliar, en caso de que los operadores de mutación y cruce no puedan generar un
        individuo válido pasado el número máximo de iteraciones, forzar la creación de uno
        que sí que lo sea.
        '''
        ### POSIBLE MEJORA: por ejemplo en cromosoma con counts=[10,1,7] pasar alguno de los 
        # 7 atributos en consecuente al antecedente, en lugar de quitarlos de la regla.
        c_aux = Chromosome(intervalos, tipos)
        counts = c_aux.count_types()
        #print(counts)
        if counts[1]==0:
            idx = random.choice([i for i, x in enumerate(tipos) if x == 0])
            tipos[idx] = 1 # No actualizamos counts porque no hace falta
        elif counts[1]>Chromosome.MAX_PER_TYPE[0]:
            while counts[1] > Chromosome.MAX_PER_TYPE[0]:
                idx = random.choice([i for i, x in enumerate(tipos) if x == 0])
                tipos[idx] = 0
                counts[1]-=1
                counts[0]+=1

        if counts[2]==0:
            idx = random.choice([i for i, x in enumerate(tipos) if x == 0])
            tipos[idx] = 2
        elif counts[2]>Chromosome.MAX_PER_TYPE[1]:
            while counts[2] > Chromosome.MAX_PER_TYPE[1]:
                idx = random.choice([i for i, x in enumerate(tipos) if x == 0])
                tipos[idx] = 0
                counts[2]-=1
                counts[0]+=1
        #print(counts)
        return Chromosome(intervalos, tipos)

    def count_types(self):
        contador = [0, 0, 0]  # Inicializamos un contador para tipos en antecedente y consecuente (resto no aparecen en regla)
        for t in self.types:
            if t not in [0, 1, 2]:
                raise ValueError("Las tipos solo pueden ser 0, 1 o 2")
            if t in [0, 1,2]:
                contador[t]+=1
        return contador
    
    # def calculate_support(self, dataset):
    #     '''
    #     Mediante la definición de esta función, guardamos el soporte de una regla de asociación (cromosoma)
    #     como un atributo propio de la clase Cromosoma, para evitar recalcular constantemente este valor.
    #     '''
    #     intervals = self.intervals
    #     types = self.types
    #     return Metrics.calculate_support(dataset, intervals, types) if dataset is not None else []


    # Función de evaluación
    def chromosome_eval(self):
        return Metrics.fitness(self)
    
    # Función de evaluación
    def chromosome_eval_multi(self):
        return Metrics.fitness_multi(self) 
    
 
'''    
    def validate(self, types):
        """
        Comprobación de que el cromosoma cumple las restricciones que se establecen en el paper de QARGA seccion 2.3 (i.e. 
        representa una regla de asociación con sentido).
        """
        antecedents = [t for t in types if t == 1]
        consequents = [t for t in types if t == 2]
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
#print("Segundo nivel del cromosoma de ejemplo:", cromosoma_ejemplo.types)
#print("Contador por cada tipo de transacción: ", cromosoma_ejemplo.counter_transaction_type)
#print("Aptitud del cromosoma:", aptitud_ejemplo)

#poblacion = toolbox.population()
#print(poblacion)
'''