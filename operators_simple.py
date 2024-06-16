import random
#from deap import base, creator, tools
from chromosome_simple import Chromosome
from dataset import Dataset
import pandas as pd

###### CONSTANTES A DEFINIR ########
#MUTATION_TYPE_PROB = 0.33 # Probabilidad de mutacion de tipo de transaccion en cromosoma
#MUTATION_INTERVAL_PROB = 0.33 # Probabilidad de mutacion en extremos del intervalo
#MIN_MAX_LIST = [] # Lista con los valores maximo y minimo por atributo - NO SE PUEDEN REBASAR
#MAX_PER_TYPE = [2,2] # Lista con el número de atributos máximo que queremos en antecedente y consecuente

class Operators:


    @staticmethod
    def crossover(ind1, ind2):
        '''
        Operador cruce tal y como aparece descrito en el paper, ligeramente modificado para devolver dos hijos
        en lugar de uno. El proceso del cruce es como sigue:
        Para cada uno de los 'genes' del cromosoma:
        - Si el tipo de transaccion en el gen para los dos padres es el mismo (i.e. el atributo correspondiente
        esta en los dos padres en antecedente, consecuente o no esta en la regla de asociacion), se selecciona
        aleatoriamente un valor de entre los extremos inferiores del intervalo de los padres, y otro para el 
        extremo superior, y posteriormente  se ordenan para que se devuelva un resultado con sentido (un 
        intervalo del tipo [l_i, u_i] donde l_i < u_i). Al hijo segundo se asignan los dos extremos restantes
        (ordenandolos tambien de manera identica).
        - Si el tipo de transaccion en el gen es distinto, a un hijo se le asignara el tipo y los intervalos
        del primer padre, y al otro el tipo y los intervalos del segundo padre.     
 
        '''
        intervalos1 = []
        transacciones1 = []
        intervalos2 = []
        transacciones2 = []
        n = len(ind1.types)
        #for _ in range(MAX_ATTEMPTS):
        for i in range(n):
            if ind1.types[i] == ind2.types[i]:
                lower_z1 = random.choice([ind1.intervals[2*i], ind2.intervals[2*i]])
                upper_z1 = random.choice([ind1.intervals[2*i+1], ind2.intervals[2*i+1]])
                lower_z2 = ind1.intervals[2*i] if lower_z1 == ind2.intervals[2*i] else ind2.intervals[2*i]
                upper_z2 = ind1.intervals[2*i+1] if upper_z1 == ind2.intervals[2*i+1] else ind2.intervals[2*i+1]
                int1 = [lower_z1, upper_z1] if lower_z1 < upper_z1 else [upper_z1, lower_z1]
                int2 = [lower_z2, upper_z2] if lower_z2 < upper_z2 else [upper_z2, lower_z2]

                intervalos1.extend(int1)
                transacciones1.append(ind1.types[i])
                intervalos2.extend(int2)
                transacciones2.append(ind2.types[i])
            else:
                t_z1 = random.choice([ind1.types[i], ind2.types[i]])
                t_z2 = ind1.types[i] if t_z1 == ind2.types[i] else ind2.types[i]
                if t_z1 == ind1.types[i]:
                    lower_z1 = ind1.intervals[2*i]
                    upper_z1 = ind1.intervals[2*i+1]
                    lower_z2 = ind2.intervals[2*i]
                    upper_z2 = ind2.intervals[2*i+1]
                else:
                    lower_z1 = ind2.intervals[2*i]
                    upper_z1 = ind2.intervals[2*i+1]
                    lower_z2 = ind1.intervals[2*i]
                    upper_z2 = ind1.intervals[2*i+1]
                intervalos1.extend([lower_z1,  upper_z1])
                transacciones1.append(t_z1)
                intervalos2.extend([lower_z2, upper_z2])
                transacciones2.append(t_z2)

        if Operators.check_valid_chromosome(transacciones1) and Operators.check_valid_chromosome(transacciones2):
            return (Chromosome(intervalos1, transacciones1),Chromosome(intervalos2, transacciones2))
        return Operators.crossover(ind1,ind2)

    @staticmethod
    def mutation(ind):
        # Mutación tipo
        #print(f"Cromosoma sin mutar: {ind}\n")
        # print("Intervalos: ", ind.intervals)
        # print("Transacciones: ", ind.types)

        for i in range(len(ind.types)):
            #if random.random() < MUTATION_TYPE_PROB:
            t_i  = ind.types[i]
            # print("Mutación en el gen: ", i)
            if t_i == 0:
                #print(ind.counter_types)
                pmt = Operators.possible_mutation_types(ind.counter_types)
                t_i = random.choice(pmt)
                ind.counter_types[t_i] += 1 # El contador de número de tipos del cromosoma sube
                ind.counter_types[0]-=1
            elif t_i == 1:
                pmt = Operators.possible_mutation_types(ind.counter_types)
                ind.counter_types[t_i] -= 1 # El contador de número de tipos del cromosoma baja
                t_i = random.choice(pmt)
                ind.counter_types[t_i] += 1
                
            ind.types[i] = t_i

            #if random.random() < MUTATION_INTERVAL_PROB:
            ### MEJORAR, numero aleatorio entre 0 y .1
            dif = random.uniform(0, 0.1)*(ind.intervals[2*i+1]-ind.intervals[2*i])
            ls_sign = Operators.check_boundaries(ind, i, dif)
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
        #print("Cromosoma mutado: \n")
        # print("Intervalos: ", ind.intervals)
        # print("Transacciones: ", ind.types)
        #ind.types = Operators.check_valid_chromosome(ind.types)
        if Operators.check_valid_chromosome(ind.types):
            #print(f"Cromosoma mutado: {ind}\n")
            return ind,
        else:
            return Operators.mutation(ind)
    
    @staticmethod
    def check_boundaries(ind, i, dif):
        ''''
        Checkea que al realizar el cambio de extremos en el intervalo del atributo,
        no nos salimos del intervalo deseado.
        '''
        min_max_ls = Operators.calculate_ranges()
        #print("min max list:", min_max_ls)
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
    def possible_mutation_types(ls_counts):
        """
        Dado el parámetro MAX_PER_TYPE, devuelve una lista con los valores posibles para la mutación de
        la transacción, asegurándose así que no se superan los límites establecidos de atributos en el
        antecedente y en el consecuente.        
        """
        if ls_counts[2] < Chromosome.MAX_PER_TYPE[1]:
            res = [2]
        elif ls_counts[1] < Chromosome.MAX_PER_TYPE[0]:
            res = [1]
        else:
            res = [0]
        return res
    
    def calculate_ranges():
        ls = []
        for c in Dataset.column_ranges:
            if Dataset.column_ranges[c]['type']=='Quantitative':
                ls.append(Dataset.column_ranges[c]['min'])
                ls.append(Dataset.column_ranges[c]['max'])
        return ls
    
    def check_valid_chromosome(types):
        '''
        Método que se asegura de que todo cromosoma sea válido y tenga 'sentido', i.e.
        que tenga al menos un atributo en antecedente y otro en consecuente.
        '''
        # Ensure at least one 1 and one 2 in the types list
        antecedents = [t for t in types if t == 1]
        consequents = [t for t in types if t == 2]
        if len(antecedents) < 1 or len(antecedents) > Chromosome.MAX_PER_TYPE[0] or len(consequents) < 1 or len(consequents) > Chromosome.MAX_PER_TYPE[1]:
            return False
        return True

'''
    NO HACE FALTA CON LA NUEVA CLASE DATASET
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