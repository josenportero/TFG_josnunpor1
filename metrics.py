from deap import base, creator, tools
#from chromosome import Chromosome # para los tests
import pandas as pd
import numpy as np
import time # Para pruebas
from dataset import Dataset


class Metrics:
    # sup, conf, cf, recov, ampl, nAttrib
    W=[1., 0.005 , 0.005 , 0.4, 0.0] #0.59 originalmente

    recov = [0 for _ in range(Dataset.dataframe.shape[0])]
    HOF = tools.HallOfFame(5)
    t_np = 0.
    t = 0.

    #n_support = 0

    def calculate_support(individual_values, individual_attribute_types):
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
        #inicio = time.time()

        support_ant = 0
        support_cons = 0
        support_rule = 0
        support = []
        #print(data)
        # Iteramos sobre las instancias
        for i in Dataset.dataframe.index:
            verifyAnt = [] 
            verifyCons = []
            # Por columna, verificamos si el valor de dicha instancia está en el rango dado por el individuo
            for c in range(len(Dataset.dataframe.columns)):
                # Comprobamos si el valor en la columna del dataframeestá en el rango especificado por el Cromosoma
                #print(c)
                #print(individual_values[c*2], " + ", individual_values[2*c+1])
                if (Dataset.dataframe.iloc[i,c] >= individual_values[c*2]) & (Dataset.dataframe.iloc[i,c] <= individual_values[c*2+1]):
                    # Solo nos importa una regla cuando una transacción está en antecedente o consecuente, i.e., es distinta a 0
                    if individual_attribute_types[c] == 1:
                        # En este caso el atributo está en antecedente
                        verifyAnt.append(True)
                    elif individual_attribute_types[c] == 2:
                        # En este caso, atributo en consecuente
                        verifyCons.append(True)
                else:
                    if individual_attribute_types[c] == 1:
                        verifyAnt.append(False)
                    elif individual_attribute_types[c] == 2:
                        verifyCons.append(False)
                # Cuando verifyAnt == True para todas las columnas, el soporte del antecedente de la regla se incrementa una unidad        
            #print(verifyAnt)
            #print(verifyCons)
            if all(verifyAnt):       
                support_ant += 1
                # Si verifyCons  == True para todas las columnas el soporte del consecuente de la regla se incrementa
                if all(verifyCons):
                    support_rule += 1
            # FPara cada instancia, si verifyCons  == True para todas las columnas el consecuente se incrementa por 1
            if all(verifyCons):
                support_cons += 1

        support.append(support_ant/len(Dataset.dataframe.index))
        support.append(support_cons/len(Dataset.dataframe.index))
        support.append(support_rule/len(Dataset.dataframe.index))

        #fin = time.time()
        #tiempo_transcurrido = fin - inicio
        
        #Imprime el tiempo transcurrido
        #print(f"El método tardó {tiempo_transcurrido:.2f} segundos en ejecutarse.")
        return support
    
    def calculate_supportv2(individual_values, individual_attribute_types):
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
        # Convertir el DataFrame a una matriz numpy para mejorar la velocidad de acceso
        # time1 = time.time()
        #Metrics.n_support +=1 
        df_np = Dataset.dataframe.to_numpy()
        
        # Inicializar matrices booleanas para antecedente y consecuente
        verifyAnt = np.ones(df_np.shape[0], dtype=bool)
        verifyCons = np.ones(df_np.shape[0], dtype=bool)

        # Iterar sobre cada columna y aplicar condiciones
        for c in range(df_np.shape[1]):
            if individual_attribute_types[c] == 1:  # Antecedente
                verifyAnt &= (df_np[:, c] >= individual_values[c*2]) & (df_np[:, c] <= individual_values[c*2+1])
            elif individual_attribute_types[c] == 2:  # Consecuente
                verifyCons &= (df_np[:, c] >= individual_values[c*2]) & (df_np[:, c] <= individual_values[c*2+1])

        # Calcular soporte
        support_ant = np.sum(verifyAnt) / df_np.shape[0]
        support_cons = np.sum(verifyCons) / df_np.shape[0]
        support_rule = np.sum(verifyAnt & verifyCons) / df_np.shape[0]

        # fin1 = time.time()
        # print(f'Ha tardado {fin1-time1} segundos')
        return [support_ant, support_cons, support_rule]

    def calculate_confidence( chromosome):
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
        soportes = chromosome.support
        return soportes[2]/soportes[0] if soportes[0]!=0. else 0.

    def calculate_lift(chromosome):
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
        soportes = chromosome.support
        return soportes[2]/(soportes[0]*soportes[1]) if (soportes[0]!=0.) & (soportes[1]!=0.) else 0.

    def covered_by_rule(individual_values, individual_attribute_types):
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
        for i in range(Dataset.dataframe.shape[0]):
            res = True
            for c in range(len(Dataset.dataframe.columns)):
                if individual_attribute_types[c] != 0:
                    value = Dataset.dataframe.iloc[i, c]
                    lower_bound = individual_values[c * 2]
                    upper_bound = individual_values[c * 2 + 1]
                    if (lower_bound > value) or (upper_bound < value):
                        res = False
                        break
            covered.append(res)
        return covered

    def measure_recovered(rules):
        """
        Calcula las regiones recuperadas por las reglas en un conjunto de datos.
        ------------------------------------------------------------------------
        Entradas:
        - dataset: DataFrame que contiene el conjunto de datos.
        - rules: Lista de cromosomas (población) que representan las reglas de asociación.

        Retorna:
        - agg: Lista de int que indica si cada instancia del dataset está cubierta por alguna regla
        de las contenidas en la lista.
        """
        n = Dataset.dataframe.shape[0]
        agg = [0 for _ in range(n)]
        if rules is not None:
            for rule in rules:
                cov = Metrics.covered_by_rule(rule.intervals, rule.types)
                agg = [x+y for x,y in zip(cov, agg)]
        #agg_plus = Metrics.covered_by_rule(chromosome.intervals, chromosome.types)
        #res = [x or y for x,y in zip(agg_plus, agg)]
        #print(agg)
        #recov_by_hof = sum([1 for i in range(n) if agg[i]>1])
        #print(f'Recubiertos por el hof: {recov_by_hof}')
        return agg if rules is not None else 0.
    
    def average_amplitude(chromosome):
        """
        Calcula la amplitud media de los atributos que aparecen en una determinada regla.
        """
        agg = [0 for i in range(len(chromosome.types))]
        cont = 0
        rang = Metrics.calculate_ranges()
        for i in range(len(chromosome.types)):
            if chromosome.types[i]!=0:
                upper = chromosome.intervals[2*i+1]
                lower = chromosome.intervals[2*i]

                minim = rang[2*i]
                maxim = rang[2*i+1]
                agg[i] = (upper - lower)/(maxim-minim)
                cont +=1
        #print(agg)
        return sum(agg)/cont if cont != 0. else 0.
        
    def calculate_ranges():
        ls = []
        for c in Dataset.column_ranges:
            if Dataset.column_ranges[c]['type']=='Quantitative':
                ls.append(Dataset.column_ranges[c]['min'])
                ls.append(Dataset.column_ranges[c]['max'])
        return ls

    def calculate_certainty_factor(chromosome):
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
        sup = chromosome.support
        sup_Y = sup[1]
        conf_XY = sup[2]/sup[0] if sup[0]!=0. else 0.
        #conf_XY = Metrics.calculate_confidence(data, individual_values, individual_attribute_types)
        #sup_Y = Metrics.calculate_support(data, individual_values, individual_attribute_types)[2]
        if conf_XY > sup_Y:
            cert = (conf_XY - sup_Y)/(1-sup_Y)
        elif conf_XY < sup_Y:
            cert = (conf_XY - sup_Y)/(sup_Y)
        else:
            cert = 0.
        return cert

    def calculate_leverage(chromosome):
        return chromosome.support[2]-chromosome.support[1]*chromosome.support[0]
    
    def calculate_gain(chromosome):
        return Metrics.calculate_confidence(chromosome)-chromosome.support[1]
    
    def calculate_chi_squared(chromosome):
        """
        Chi2 alto: mayor probabilidad dependencia entre las variables
        Chi2 bajo: mayor probabilidad de independencia
        """
        support = chromosome.support
        n = len(Dataset.dataframe.index)
        #print(O_11)
        conf = Metrics.calculate_confidence(chromosome)
        lift = Metrics.calculate_lift(chromosome)
        chi2 = n*np.power((lift-1),2)*(support[2]*conf)/((conf-support[2])*(lift-conf)) if conf != support[2] and (lift != conf) else float('inf')

        return chi2

    # def pseudo_recov(rules, rule):
    #     """
    #     Calcula el solapamiento normalizado de una regla con respecto a un conjunto de reglas existentes.
        
    #     Entradas:
    #     - rules: Lista de reglas existentes.
    #     - rule: La nueva regla que se está evaluando.
        
    #     Salida:
    #     - overlap_normalizado: Solapamiento normalizado de la nueva regla con respecto al conjunto de reglas existentes.
    #     """
    #     total_overlap = 0
    #     for r in rules:
    #         for i in range(len(rule.types)):
    #             if r.types[i] == rule.types[i] and rule.types[i] != 0:  # Consider only if types match and are not zero
    #                 overlap, overlap_length = Metrics.interval_overlap(
    #                     [r.intervals[2*i], r.intervals[2*i+1]], 
    #                     [rule.intervals[2*i], rule.intervals[2*i+1]]
    #                 )
    #                 if overlap:
    #                     interval_total_length = max(r.intervals[2*i+1], rule.intervals[2*i+1]) - min(r.intervals[2*i], rule.intervals[2*i])
    #                     total_overlap += overlap_length / interval_total_length
    
    #     return total_overlap / len(rule.types)  # Normalize by the number of attributes

    # def interval_overlap(interval1, interval2):
        """
        Calcula si dos intervalos se solapan y cuánto se solapan.
        
        Parámetros:
        - interval1: Lista de tamaño 2 que representa el primer intervalo [a, b].
        - interval2: Lista de tamaño 2 que representa el segundo intervalo [c, d].
        
        Retorna:
        - Tuple: (solapan, longitud_solapamiento)
            - solapan: Booleano que indica si los intervalos se solapan.
            - longitud_solapamiento: La longitud del solapamiento (0 si no se solapan).
        """
        a, b = interval1
        c, d = interval2
        
        # Verificar si se solapan
        if a <= d and c <= b:
            # Calcular la longitud del solapamiento
            overlap_length = min(b, d) - max(a, c)
            return True, overlap_length
        else:
            return False, 0

    def fitness(chromosome):
        """
        Cálculo de la función fitness tal y como aparece en el paper 'QARGA'
        """
        if len(chromosome.support)==0:
            #inicio =time.time()
            #aux = Metrics.calculate_support(chromosome.intervals, chromosome.types)
            #fin = time.time()
            #Metrics.t += fin-inicio
            #inicio2 =time.time()
            aux2 = Metrics.calculate_supportv2(chromosome.intervals,chromosome.types)
            #fin2 = time.time()
            #Metrics.t_np += fin2-inicio2
            chromosome.support = aux2
        else:
            aux2 = chromosome.support
        sup = aux2[2]
        conf = aux2[2]/aux2[0] if aux2[0]!=0. else 0.
        #lift = aux2[2]/(aux2[0]*aux2[1]) if (aux2[0]!=0.) & (aux2[1]!=0.) else 0.
  
        cf = Metrics.calculate_certainty_factor(chromosome)

        n = Dataset.dataframe.shape[0]
        recovered_by_rule = Metrics.measure_recovered([chromosome])
        recubiertos = sum([1 for i in range(n) if Metrics.recov[i] + recovered_by_rule[i] > 1])/n
        #pseudo_recov = Metrics.pseudo_recov(Metrics.HOF, chromosome)
        #print(f'Recubiertos={recubiertos}')
        #conviction = (1-chromosome.support[1])/(1-conf)  if conf!=1. else float('inf')
        #nAttrib = sum([e for i,e in enumerate(chromosome.counter_types) if i>0])
        #ampl = Metrics.average_amplitude(chromosome)
        #print('nAttrib: ',nAttrib)
        #chisq = Metrics.calculate_chi_squared(chromosome)
        #print(ampl)
        return Metrics.W[0]*sup+Metrics.W[1]*conf+Metrics.W[2]*cf-Metrics.W[3]*recubiertos,
        # return Metrics.W[0]*sup + Metrics.W[1]*conf + Metrics.W[2]*cf - Metrics.W[3]*recubiertos,#Metrics.W[4]*Metrics.average_amplitude(chromosome),

    def fitness_multi(chromosome):
        """
        Cálculo de la función fitness tal y como aparece en el paper 'QARGA'
        """
        if len(chromosome.support)==0:
            aux = Metrics.calculate_supportv2(chromosome.intervals, chromosome.types)
            chromosome.support = aux
        else:
            aux = chromosome.support
        sup = aux[2]
        conf = aux[2]/aux[0] if aux[0]!=0. else 0.
        #lift = aux[2]/(aux[0]*aux[1]) if (aux[0]!=0.) & (aux[1]!=0.) else 0.

        lev = sup - aux[1]*aux[0]
        #lift = Metrics.calculate_lift(chromosome)
        #gain = Metrics.calculate_gain(chromosome)
        # cf = Metrics.calculate_certainty_factor(chromosome)
        #conviction = (1-chromosome.support[1])/(1-conf)  if conf!=1. else float('inf')
        #n = Dataset.dataframe.shape[0]
        #recovered_by_rule = Metrics.measure_recovered([chromosome])
        #recubiertos = sum([1 for i in range(n) if Metrics.recov[i] + recovered_by_rule[i] > 1])/n
        #nAttrib = sum([e for i,e in enumerate(chromosome.counter_types) if i>0])
        #ampl = Metrics.average_amplitude(chromosome)
        #print('nAttrib: ',nAttrib)
        #print(ampl)
        return sup, lev #(recov) #- Metrics.W[4]*ampl+Metrics.W[5]*nAttrib,
