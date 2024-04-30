import random
from deap import base, creator, tools
import pandas as pd
from metrics import Metrics


#### CONSTANTES DEFINIDAS
MAX_PER_TYPE = [2,2] # Lista con el número de atributos máximo que queremos en antecedente y consecuente

class Chromosome:
    def __init__(self, intervals=None, transactions=None):
        self.intervals = intervals if intervals else []
        self.transactions = transactions if transactions else []
        self.counter_transaction_type = self.count_transactions()

    def create_chromosome(n, min_val, max_val):
        intervalos = []
        transacciones = []
        count = [0,0]
        for _ in range(n):
            inf = random.uniform(min_val, max_val)
            sup = random.uniform(inf, max_val)
            intervalos.extend([inf, sup])
            t = random.choice([0,1,2])
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
        return Chromosome(intervalos, transacciones)

    def count_transactions(self):
        contador = [0, 0]  # Inicializamos un contador para transacciones en antecedente y consecuente (resto no aparecen en regla)
        for t in self.transactions:
            if t not in [0, 1, 2]:
                raise ValueError("Las transacciones solo pueden ser 0, 1 o 2")
            if t in [1,2]:
                contador[t-1]+=1
        return contador

    # Función de evaluación
    def chromosome_eval(self, dataset, w=[1.,1.,1.,1.,1.]):
        # TEMPORAL - Definir fitness mas claramente aqui en vez de con creator
        return Metrics.fitness(self, dataset, w)
    
    
'''
# Definiciones creator DEAP
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


# Ejemplo de uso
####### DATOS 'DE JUGUETE' PARA COMPROBACIÓN DE LAS MÉTRICAS
data = {
    'A': [1.2, 2.3, 3.4, 4.5, 5.6],
    'B': [7.8, 8.9, 9.0, 10.1, 11.2],
    'C': [13.4, 14.5, 15.6, 16.7, 17.8],
    'D': [19.0, 20.1, 21.2, 22.3, 23.4],
    'E': [25.6, 26.7, 27.8, 28.9, 30.0]
}

df = pd.DataFrame(data)


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