import random
from deap import base, creator, tools
from chromosome import Chromosome

###### CONSTANTES A DEFINIR ########
MUTATION_TYPE_PROB = 0.1 # Probabilidad de mutacion de tipo de transaccion en cromosoma
MUTATION_INTERVAL_PROB = 0.1 # Probabilidad de mutacion en extremos del intervalo
MIN_MAX_LIST = [] # Lista con los valores maximo y minimo por atributo - NO SE PUEDEN REBASAR


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
    def mutation(ind):
        # Mutación tipo
        for i in range(len(ind.transactions)):
            if random.random() < MUTATION_TYPE_PROB:
                t_i  = ind.transactions[i]
                if t_i == 0:
                    t_i = random.choice([1,2])
                else:
                    t_i = 0
                ind.transactions[i] = t_i
            if random.random() < MUTATION_INTERVAL_PROB:
                ### MEJORAR, numero aleatorio entre 0 y .1
                dif = 0.05*(ind.intervals[2*i+1]-ind.intervals[2*i])
                ls_sign = Operators.check_boundaries(ind, i)
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
    def check_boundaries(ind, i):
        ''''
        Checkea que al realizar el cambio de extremos en el intervalo del atributo,
        no nos salimos del intervalo deseado.
        '''
        dif = 0.05*(ind.intervals[2*i+1]-ind.intervals[2*i])
        if ind.intervals[2*i]-dif < MIN_MAX_LIST[2*i] and ind.intervals[2*i+1]+dif > MIN_MAX_LIST[2*i+1]:
            sign1 = +1       
            sign2 = -1
        elif ind.intervals[2*i]-dif < MIN_MAX_LIST[2*i]:
            sign1= +1
            sign2 = random.choice([1,-1])
        elif ind.intervals[2*i+1]+dif > MIN_MAX_LIST[2*i+1]:
            sign1 = random.choice([1,-1])
            sign2 = -1
        else:
            sign1 = random.choice([1,-1])
            sign2 = random.choice([1,-1])
        return [sign1, sign2]


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
print("Individuo 2 - Intervalos:", ind2.intervals)
print("Individuo 2 - Transacciones:", ind2.transactions)


ind_cruce = toolbox.mate(ind1, ind2)

print("\nDespués del cruce:")
print("Individuo Cruce - Intervalos:", ind_cruce.intervals)
print("Individuo Cruce - Transacciones:", ind_cruce.transactions)

print("\nDespués de la mutación:")
ind1 = toolbox.mutate(ind1)
ind2 = toolbox.mutate(ind2)
print("Individuo 1 - Intervalos:", ind1.intervals)
print("Individuo 1 - Transacciones:", ind1.transactions)
print("Individuo 2 - Intervalos:", ind2.intervals)
print("Individuo 2 - Transacciones:", ind2.transactions)