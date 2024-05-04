import random
from deap import base, creator, tools
from chromosome import Chromosome 

###### CONSTANTES A DEFINIR ########
MUTATION_TYPE_PROB = 0.33 # Probabilidad de mutacion de tipo de transaccion en cromosoma
MUTATION_INTERVAL_PROB = 0.33 # Probabilidad de mutacion en extremos del intervalo
MIN_MAX_LIST = [] # Lista con los valores maximo y minimo por atributo - NO SE PUEDEN REBASAR
MAX_PER_TYPE = [2,2] # Lista con el número de atributos máximo que queremos en antecedente y consecuente

class Operators:
    @staticmethod
    def crossover(ind1, ind2):
        # Operador cruce descrito en el paper
        intervalos = []
        transacciones = []
        n = len(ind1.transactions)
        for i in range(n):
            if ind1.transactions[i] == ind2.transactions[i]:
                lower_z = random.choice([ind1.intervals[2*i], ind2.intervals[2*i]])
                if (ind1.intervals[2*i+1] > lower_z) & (ind2.intervals[2*i+1] > lower_z):
                    upper_z = random.choice([ind1.intervals[2*i+1], ind2.intervals[2*i+1]])
                else:
                    upper_z = max([ind1.intervals[2*i+1], ind2.intervals[2*i+1]])
                intervalos.extend([lower_z, upper_z])
                transacciones.append(ind1.transactions[i])
            else:
                t_z = random.choice([ind1.transactions[i], ind2.transactions[i]])
                if t_z == ind1.transactions[i]:
                    lower_z = ind1.intervals[2*i]
                    upper_z = ind1.intervals[2*i+1]
                else:
                    lower_z = ind2.intervals[2*i]
                    upper_z = ind2.intervals[2*i+1]
                intervalos.extend([lower_z,  upper_z])
                transacciones.append(t_z)
        return Chromosome(intervalos, transacciones)

    @staticmethod
    def mutation(ind, dataset):
        # Mutación tipo
        for i in range(len(ind.transactions)):
            if random.random() < MUTATION_TYPE_PROB:
                t_i  = ind.transactions[i]
                #print("Mutación en el gen: ", i)
                if t_i == 0:
                    #print(ind.counter_transaction_type)
                    pmt = Operators.possible_mutation_types(ind.counter_transaction_type, dataset)
                    t_i = random.choice(pmt)
                    ind.counter_transaction_type[t_i-1] += 1 # El contador de número de transacciones del cromosoma sube
                else:
                    ind.counter_transaction_type[t_i-1] -= 1 # El contador de número de transacciones del cromosoma baja
                    t_i = 0    
                ind.transactions[i] = t_i
            if random.random() < MUTATION_INTERVAL_PROB:
                ### MEJORAR, numero aleatorio entre 0 y .1
                print("true")
                dif = 0.05*(ind.intervals[2*i+1]-ind.intervals[2*i])
                ls_sign = Operators.check_boundaries(ind, i, dataset)
                sign1 = ls_sign[0]
                sign2 = ls_sign[1]
                tipo_mut = random.choice([0,1,2])
                if tipo_mut == 0:
                    ind.intervals[2*i] = ind.intervals[2*i]+sign1*dif
                elif tipo_mut == 1:
                    ind.intervals[2*i+1] = ind.intervals[2*i+1]+sign2*dif
                else:
                    ind.intervals[2*i] = ind.intervals[2*i]+sign1*dif
                    ind.intervals[2*i+1] = ind.intervals[2*i+1]+sign2*dif
        return ind
    
    @staticmethod
    def check_boundaries(ind, i, dataset):
        ''''
        Checkea que al realizar el cambio de extremos en el intervalo del atributo,
        no nos salimos del intervalo deseado.
        '''
        min_max_ls = Operators.calculate_ranges(dataset)
        #print("min max list:", min_max_ls)
        dif = 0.05*(ind.intervals[2*i+1]-ind.intervals[2*i])
        if ind.intervals[2*i]-dif < min_max_ls[2*i] and ind.intervals[2*i+1]+dif > min_max_ls[2*i+1]:
            sign1 = +1       
            sign2 = -1
        elif ind.intervals[2*i]-dif < min_max_ls[2*i]:
            sign1= +1
            sign2 = random.choice([1,-1])
        elif ind.intervals[2*i+1]+dif > min_max_ls[2*i+1]:
            sign1 = random.choice([1,-1])
            sign2 = -1
        else:
            sign1 = random.choice([1,-1])
            sign2 = random.choice([1,-1])
        return [sign1, sign2]
    
    @staticmethod
    def possible_mutation_types(ls_count_transactions, dataset):
        """
        Dado el parámetro MAX_PER_TYPE, devuelve una lista con los valores posibles para la mutación de
        la transacción, asegurándose así que no se superan los límites establecidos de atributos en el
        antecedente y en el consecuente.        
        """
        if (ls_count_transactions[0] < dataset.max_per_type[0]) & (ls_count_transactions[1] < dataset.max_per_type[1]):
            res = [1,2]
        elif ls_count_transactions[0] < dataset.max_per_type[0]:
            res = [1]
        elif ls_count_transactions[1] < dataset.max_per_type[1]:
            res = [2]
        else:
            res = [0]
        #print(res)
        return res
    
    def calculate_ranges(dataset):
        ls = []
        for c in dataset.column_ranges:
            if dataset.column_ranges[c]['type']=='Quantitative':
                ls.append(dataset.column_ranges[c]['min'])
                ls.append(dataset.column_ranges[c]['max'])
        return ls

'''
    def calculate_ranges(dataset):
        """
        Recibida una población inicial, este método calcula cuáles son los rangos admisibles entre los
        que se moverá cada atributo.

        Entrada:
        - population: Una lista de cromosomas (objetos Chromosome) que representan la población inicial.

        Salida:
        - lista que contiene los rangos admisibles para cada atributo en la población.
        Cada rango se representa como una tupla (límite_inferior, límite_superior).
        """
        min_max_dict = dict()  # Diccionario para almacenar los valores mínimos y máximos por columna

        for column in dataset.columns:
            # Calcula el mínimo y el máximo de la columna actual
            min_value = dataset[column].min()
            max_value = dataset[column].max()

            # Almacena los valores en el diccionario
            min_max_dict[column] = [min_value, max_value]

        return [item for sublist in list(min_max_dict.values()) for item in sublist]
'''
