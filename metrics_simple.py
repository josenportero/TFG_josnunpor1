from deap import base, creator, tools
#from chromosome import Chromosome # para los tests
import pandas as pd
import time # Para pruebas

class Metrics:

    W=[0.599,0.4,0.001,0.,0.]

    SUP_CALC = 0

    def calculate_support(data, individual_values, individual_attribute_types):
        """
        Calcula el soporte para reglas de asociación en un conjunto de datos.
        ---------------------------------------------------------------------
        Entradas:
        - data: dataset que contiene el conjunto de datos.
        - individual_values: Lista de valores individuales que definen los intervalos de los atributos en la regla.
        - individual_attribute_types: Lista de tipos de atributos (0 si no está en la regla,
          1 para antecedente, 2 para consecuente).

        Salidas:
        - support: Lista que contiene el soporte del antecedente, consecuente y regla en ese orden.
        """
        # Metrics.SUP_CALC += 1
        inicio = time.time()

        support_ant = 0
        support_cons = 0
        support_rule = 0
        support = []
        #print(data)
        # Iterate over instances
        for i in data.dataframe.index:
            verifyAnt = [] 
            verifyCons = []
            # For each column verify if the value of that instance is in the range given by the individual
            for c in range(len(data.dataframe.columns)):
                # We check that te value in the Dataframe's column is in the range specified by the Chromosome
                #print(c)
                #print(individual_values[c*2], " + ", individual_values[2*c+1])
                if (data.dataframe.iloc[i,c] >= individual_values[c*2]) & (data.dataframe.iloc[i,c] <= individual_values[c*2+1]):
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

        support.append(support_ant/len(data.dataframe.index))
        support.append(support_cons/len(data.dataframe.index))
        support.append(support_rule/len(data.dataframe.index))

        fin = time.time()
        tiempo_transcurrido = fin - inicio
        
        # #Imprime el tiempo transcurrido
        print(f"El método tardó {tiempo_transcurrido:.2f} segundos en ejecutarse.")
        return support


    def calculate_confidence(data, individual_values, individual_attribute_types):
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
        soportes = Metrics.calculate_support(data, individual_values, individual_attribute_types)
        return soportes[2]/soportes[0] if soportes[0]!=0. else 0.

    def calculate_lift(data, individual_values, individual_attribute_types):
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
        soportes = Metrics.calculate_support(data, individual_values, individual_attribute_types)
        return soportes[2]/(soportes[0]*soportes[1]) if (soportes[0]!=0.) & (soportes[1]!=0.) else 0.

    def covered_by_rule(data, individual_values, individual_attribute_types):
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
        covered = []
        for i in data.dataframe.index:
            res = True
            for c in range(len(data.dataframe.columns)):
                if individual_attribute_types[c] != 0:
                    value = data.dataframe.iloc[i, c]
                    lower_bound = individual_values[c * 2]
                    upper_bound = individual_values[c * 2 + 1]
                    if not (lower_bound <= value <= upper_bound):
                        res = False
                        break
            covered.append(res)
        return covered

    def measure_recovered(data, rules):
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
        n = data.dataframe.shape[0]
        agg = [False for _ in range(n)]
        if rules is not None:
            for rule in rules:
                cov = Metrics.covered_by_rule(data, rule.intervals, rule.transactions)
                agg = [x or y for x,y in zip(cov, agg)]
        return sum(agg)/n if rules is not None else 0.
    
    def calculate_certainty_factor(data, individual_values, individual_attribute_types):
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
        conf_XY = Metrics.calculate_confidence(data, individual_values, individual_attribute_types)
        sup_Y = Metrics.calculate_support(data, individual_values, individual_attribute_types)[2]
        if conf_XY > sup_Y:
            cert = (conf_XY - sup_Y)/(1-sup_Y)
        elif conf_XY < sup_Y:
            cert = (conf_XY - sup_Y)/(sup_Y)
        else:
            cert = 0.
        return cert
    
    def fitness(chromosome, dataset):
        """
        Cálculo de la función fitness tal y como aparece en el paper 'QARGA'
        """
        if len(chromosome.support)==0:
            aux = Metrics.calculate_support(dataset, chromosome.intervals, chromosome.transactions)
            chromosome.support = aux
        else:
            aux = chromosome.support
        sup = aux[2]
        conf = aux[2]/aux[0] if aux[0]!=0. else 0.
        lift = aux[2]/(aux[0]*aux[1]) if (aux[0]!=0.) & (aux[1]!=0.) else 0.
        #recov = Metrics.measure_recovered(dataset, hof)
        # nAttrib = sum(chromosome.counter_transaction_type)
        # grouped_ls = [[chromosome.intervals[i], chromosome.intervals[i + 1]] for i in range(0, len(chromosome.intervals), 2)]
        # agg = [0 for _ in range(len(grouped_ls))]
        # for i in range(len(grouped_ls)):
        #     if chromosome.transactions[i]!=0:
        #         agg[i] = grouped_ls[i][1]-grouped_ls[i][0]
        # non_zero_t = (len(agg)-sum(1 for i in range(len(agg)) if agg[i]==0))
        # ampl_total = sum(agg)/non_zero_t if non_zero_t != 0. else 0.
        # max_ampl = max(agg)
        # ampl = ampl_total/(max_ampl) if max_ampl != 0. else 0.
        #print('nAttrib: ',nAttrib)
        #print(ampl)
        #print(w[0]*sup + w[1]*conf + w[3]*nAttrib  -w[4]*ampl,)
        return Metrics.W[0]*sup + Metrics.W[1]*conf + Metrics.W[2]*lift,  #- Metrics.W[4]*ampl,
        
