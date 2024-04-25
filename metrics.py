from deap import base, creator, tools
from chromosome import Chromosome
import pandas as pd

class Metrics:

    def calculate_support(data, individual_values, individual_attribute_types):
        """
        Calcula el soporte para reglas de asociación en un conjunto de datos.
        ---------------------------------------------------------------------
        Entradas:
        - data: DataFrame que contiene el conjunto de datos.
        - individual_values: Lista de valores individuales que definen los intervalos de los atributos en la regla.
        - individual_attribute_types: Lista de tipos de atributos (0 si no está en la regla,
          1 para antecedente, 2 para consecuente).

        Salidas:
        - support: Lista que contiene el soporte del antecedente, consecuente y regla en ese orden.
        """
        support_ant = 0
        support_cons = 0
        support_rule = 0
        support = []
        #print(data)
        # Iterate over instances
        for i in data.index:
            verifyAnt = [] 
            verifyCons = []
            # For each column verify if the value of that instance is in the range given by the individual
            for c in range(len(data.columns)):
                # We check that te value in the Dataframe's column is in the range specified by the Chromosome
                #print(c)
                #print(individual_values[c*2], " + ", individual_values[2*c+1])
                if (data.iloc[i,c] >= individual_values[c*2]) & (data.iloc[i,c] <= individual_values[c*2+1]):
                    # We only care about a rule when transaction is different than null, i.e., transaction[c]!=0 
                    if individual_attribute_types[c] == 1:
                        # In this case, the rule is in the antecedent
                        verifyAnt.append(True)
                    elif individual_attribute_types[c] == 2:
                        # In this case, rule is in consequent
                        verifyCons.append(True)
                else:
                    if individual_attribute_types[c] == 1:
                        verifyAnt.append(False)
                    elif individual_attribute_types[c] == 2:
                        verifyCons.append(False)
            # When verifyAnt == True for all the columns, the support of the antecedent of the rule increases by 1
            #print(verifyAnt)
            #print(verifyCons)
            if all(verifyAnt): 
                support_ant += 1
                # If verifyCons  == True for all the columns the rule support increases by 1
                if all(verifyCons):
                    support_rule += 1
            # For each instance, if verifyCons  == True for all the columns the consequent support increases by 1
            if all(verifyCons):
                support_cons += 1
        support.append(support_ant/len(data.index))
        support.append(support_cons/len(data.index))
        support.append(support_rule/len(data.index))
        return support


    def calculate_confidence(dataset, individual_values, individual_attribute_types):
        """
        Calcula la confianza de una regla X => Y en un conjunto de datos.
        -----------------------------------------------------------------
        Entradas:
        - dataset: DataFrame que contiene el conjunto de datos.
        - individual_values: Lista de valores individuales que definen los intervalos de los atributos en la regla.
        - individual_attribute_types: Lista de tipos de atributos (0 si no está en la regla,
        1 para antecedente, 2 para consecuente).

        Salida:
        - confidence: Confianza en la regla X => Y ~ Probabilidad de que las reglas que contienen a X contengan,
        a su vez, a Y.
        """
        soportes = Metrics.calculate_support(dataset, individual_values, individual_attribute_types)
        return soportes[2]/soportes[0] if soportes[0]!=0. else 0.

    def calculate_lift(dataset, individual_values, individual_attribute_types):
        """
        Calcula el lift de la regla X => Y en el dataset.
        -------------------------------------------------
        Entradas:
        - dataset: DataFrame que contiene el conjunto de datos.
        - individual_values: Lista de valores individuales que definen los intervalos de los atributos en la regla.
        - individual_attribute_types: Lista de tipos de atributos (0 si no está en la regla,
        1 para antecedente, 2 para consecuente).

        Salida:
        - lift: Lift de la regla X => Y ~ Cuántas veces son más probables X e Y juntas en el dataset de lo esperado,
        asumiendo que X e Y son ocurrencias estadísticamente independientes
        """
        soportes = Metrics.calculate_support(dataset, individual_values, individual_attribute_types)
        return soportes[2]/(soportes[0]*soportes[1]) if (soportes[0]!=0.) & (soportes[1]!=0.) else 0.

    def covered_by_rule(dataset, individual_values, individual_attribute_types):
        """
        Determina qué ejemplos determinados del dataset son cubiertos o no por una regla.
        ---------------------------------------------------------------------------------
        Entradas:
        - dataset: DataFrame que contiene el conjunto de datos.
        - individual_values: Lista de valores individuales que definen los intervalos de los atributos en la regla.
        - individual_attribute_types: Lista de tipos de atributos (0 si no está en la regla,
        1 para antecedente, 2 para consecuente).

        Salida:
        - covered: Lista de booleanos que indica si cada instancia del dataset está cubierta por la regla.
        """
        cov=[]
        for i in dataset.index:
            # For each column verify if the value of that instance is in the range given by the individual
            for c in range(len(dataset.columns)-1):
                res=True
                if individual_attribute_types[c]!=0:
                    res=res&(dataset.iloc[i,c] >= individual_values[c*2]) & (dataset.iloc[i,c] <= individual_values[c*2+1])
            cov.append(res)
        return cov

    def measure_recovered(dataset, rules):
        """
        Calcula las regiones recuperadas por las reglas en un conjunto de datos.
        ------------------------------------------------------------------------
        Entradas:
        - dataset: DataFrame que contiene el conjunto de datos.
        - rules: Lista de cromosomas (población) que representan las reglas de asociación.

        Retorna:
        - agg: Lista de booleanos que indica si cada instancia del dataset está cubierta por alguna regla
        de las contenidas en la lista.
        """
        agg = [False for _ in range(dataset.shape[0])]
        for rule in rules:
            cov = Metrics.covered_by_rule(dataset, rule.intervals, rule.transactions)
            agg = [x or y for x,y in zip(cov, agg)]
        return agg
    
    def calculate_certainty_factor(dataset, individual_values, individual_attribute_types):
        """
        Calcula el factor de certeza para una regla X => Y en un conjunto de datos.

        Entradas:
        - dataset: DataFrame que contiene el conjunto de datos.
        - individual_values: Lista de valores individuales que definen los intervalos de los atributos en la regla.
        - individual_attribute_types: Lista de tipos de atributos (0 si no está en la regla,
        1 para antecedente, 2 para consecuente).

        Salida:
        - cert: Valor del factor de certeza para la regla X => Y.
        """
        cert = 0.
        conf_XY = Metrics.calculate_confidence(dataset, individual_values, individual_attribute_types)
        sup_Y = Metrics.calculate_support(dataset, individual_values, individual_attribute_types)[2]
        if conf_XY > sup_Y:
            cert = (conf_XY - sup_Y)/(1-sup_Y)
        elif conf_XY < sup_Y:
            cert = (conf_XY - sup_Y)/(sup_Y)
        else:
            cert = 0.
        return cert


####### DATOS 'DE JUGUETE' PARA COMPROBACIÓN DE LAS MÉTRICAS
data = {
    'A': [1.2, 2.3, 3.4, 4.5, 5.6],
    'B': [7.8, 8.9, 9.0, 10.1, 11.2],
    'C': [13.4, 14.5, 15.6, 16.7, 17.8],
    'D': [19.0, 20.1, 21.2, 22.3, 23.4],
    'E': [25.6, 26.7, 27.8, 28.9, 30.0]
}

df = pd.DataFrame(data)

# Creación de población de cromosomas para comprobar las métricas
population = [Chromosome.create_chromosome(5, 0, 30) for _ in range(10)]  # Creamos 1 cromosomas de ejemplo

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

print('\n======================================')
print("'Medida recuperada' por las reglas: ", Metrics.measure_recovered(df, population))
