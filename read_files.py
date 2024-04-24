import csv
import pandas as pd
import datetime as dt
import functools as f
import numpy as np


from datetime import timedelta, datetime

################################################
# Rutas de los archivos con datos a considerar #
################################################
datosDemanda = "datos/export_DemandaReal_2022_2023.json"
datosPrecio = "datos/export_PrecioMedioHorarioFinalSumaDeComponentes_2022_2023.json"
datosGeneracionSolar = "datos/export_GeneracionTRealSolar_2022_2023.json"
datosGeneracionEolica = "datos/export_GeneracionTRealEolica_2022_2023.json"
datosGas = "datos/MIBGAS_Data_2022_2023.csv"
datosTempBarcelona = "datos/tempBarcelona_2022_2023.json"
datosTempValencia = "datos/tempValencia_2022_2023.json"
datosTempMadrid = "datos/tempMadrid_2022_2023.json"
datosTempSevilla= "datos/tempSevilla_2022_2023.json"


def obtener_tabla_datos(datosDemanda, datosPrecio, datosGeneracionSolar, datosGeneracionEolica,
                        datosGas, datosTempBarcelona, datosTempValencia, datosTempMadrid,
                        datosTempSevilla):
    '''
    Devuelve dataframe final con todos los datos de la serie temporal que se va a analizar,
    concatenando los distintos dataframes obtenidos a partir de las rutas de los ficheros
    que se le han pasado como argumentos.
    -------------------------------------------------------------------------------------
    '''
    df1 = procesar_datos_demanda(datosDemanda)
    df2 = procesar_datos_GeneracionSolar(datosGeneracionSolar)
    df3 = procesar_datos_GeneracionEolica(datosGeneracionEolica)
    df4 = procesar_datos_gas(datosGas)
    df5 = procesar_datos_temperatura(datosTempBarcelona)
    df6 = procesar_datos_temperatura(datosTempMadrid)
    df7 = procesar_datos_temperatura(datosTempSevilla)
    df8 = procesar_datos_temperatura(datosTempValencia)
    df9 = procesar_datos_precio(datosPrecio)

    # Concatenacion de los datagramas por filas
    df_final = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1)

    # Exportamos el dataframe resultante a un Excel para ver los resultados
    df_final.to_excel('intento.xlsx',index=False)
    return print(df_final.head())

def procesar_datos_demanda(datosDemanda):
    """
    Procesa los datos sobre la demanda de energia electrica contenidos en un archivo json.
    ---------------------------------------------------------
    Datos: ruta del archivo json que contiene los datos sobre la demanda real.
    Devuelve: Dataframe pandas que contiene los datos de demanda real por fecha,
    hora y día de la semana (estos dos últimos codificados cíclicamente) después
    de cambiar el nombre de la columna "value" a "demanda".
    """
    df = pd.read_json(datosDemanda)
    # Analizamos precios en España (UTC+1), +1:00 es informacion redundante
    df['datetime']=df['datetime'].str.split('+').str[0]
    df.rename(columns={"values": "demanda", "datetime":"fecha"}, inplace=True)

    # Convertimos la columna "fecha" al tipo datetime
    df["fecha"] = pd.to_datetime(df["fecha"], utc=False)

    # Agregamos una nueva columna con el día de la semana (1 para lunes, 2 para martes, etc.)
    df['diaSemana'] = df['fecha'].dt.dayofweek + 1

    # Añadimos una nueva columna 'hora' separando la columna fecha en dos
    df['hora']= df['fecha'].dt.hour
    df['fecha'] = df['fecha'].dt.date #----> para la comprobación de fechas ausentes

    #Codificacion ciclica de las variables diaSemana y hora
    hora_ciclico = codificacion_ciclica(df['hora'],longitud_ciclo=24)
    dia_semana_ciclico = codificacion_ciclica(df['diaSemana'], longitud_ciclo=7)

    # Nos quedamos solo con las columnas que nos interesan
    cols = ["fecha","demanda"]
    df_demanda = pd.concat([df[cols], hora_ciclico, dia_semana_ciclico], axis=1)

    return df_demanda

def procesar_datos_precio(datosPrecio):
    """
    Procesa los datos de precios contenidos en un archivo json.
    ---------------------------------------------------------
    Datos: ruta del archivo json que contiene los datos de precios.
    Devuelve: Dataframe pandas que contiene los datos de precios después de cambiar el nombre de la columna a "precio".
    """

    df = pd.read_json(datosPrecio)
    df.rename(columns={"values": "precio", "datetime":"fecha"}, inplace=True)  # Cambia el nombre de la columna en el DataFrame original
    
    #### NO USADO EN VERSION FINAL
    # Analizamos datos en España (UTC+1), +1:00 es informacion redundante
    #df['fecha']=df['fecha'].str.split('+').str[0]

    # Convertimos la columna "fecha" al tipo datetime
    #df["fecha"] = pd.to_datetime(df["fecha"], utc=False).dt.date

    cols=["precio"]
    df_prec = df[cols]  # Extrae la columna renombrada
    
    return df_prec

def procesar_datos_GeneracionSolar(datosGeneracionSolar):
    """
    Procesa los datos de generacion solar contenidos en un archivo json.
    ---------------------------------------------------------
    Datos: ruta del archivo json que contiene los datos de generacion solar.
    Devuelve: Dataframe pandas que contiene los datos de generacion solar
        después de cambiar el nombre de la columna a "generacionSolar".
    """

    df = pd.read_json(datosGeneracionSolar)
    df.rename(columns={"values": "generacionSolar", "datetime":"fecha"}, inplace=True)  # Cambia el nombre de la columna en el DataFrame original
    
    #### NO USADO EN VERSION FINAL
    # Analizamos datos en España (UTC+1), +1:00 es informacion redundante
    #df['fecha']=df['fecha'].str.split('+').str[0]

    # Convertimos la columna "fecha" al tipo datetime
    #df["fecha"] = pd.to_datetime(df["fecha"], utc=False).dt.date

    cols=[ "generacionSolar"]
    df_gs = df[cols]  # Extrae la columna renombrada

    return df_gs

def procesar_datos_GeneracionEolica(datosGeneracionEolica):
    """
    Procesa los datos de generacion eolica contenidos en un archivo json.
    ---------------------------------------------------------
    Datos: ruta del archivo json que contiene los datos de generacion eolica.
    Devuelve: Dataframe pandas que contiene los datos de generacion eolica
        después de cambiar el nombre de la columna a "generacionEolica".
    """
    df = pd.read_json(datosGeneracionEolica)
    df.rename(columns={"values": "generacionEolica", "datetime":"fecha"}, inplace=True)  # Cambia el nombre de la columna en el DataFrame original

    # Convertimos la columna "fecha" al tipo datetime // columna no usada de momento
    #df['fecha'] = pd.to_datetime(df['fecha'], utc=True).dt.tz_convert('Europe/Madrid')

    cols=["generacionEolica"]
    df_ge = df[cols]  # Extrae la columna renombrada
    
    
    return df_ge

def procesar_datos_gas(datos):
    """
    Procesa los datos sobre el precio del gas contenidos en un archivo CSV.
    ---------------------------------------------------------   
    Datos: ruta del archivo CSV que contiene los datos de precio del gas.
    Devuelve: Dataframe pandas que contiene los datos de precio del gas
        después de cambiar el nombre de la columna a "precioGas".

    Nota 1: unicamente se incluyen datos sobre el producto 'GDAES_D+1', puesto
    que solo de este se tienen datos diarios
    Nota 2: los precios de gas son diarios (no por horas) asi que al incluirlos
    en el dataframe final cada valor debera ser repetido 24 veces 
    """
    df = pd.read_csv(datos, skiprows=[0], sep=';') # La primera fila no contiene informacion util
    df_filtrado = df[df["Product"]=="GDAES_D+1"]
    df_aux=df_filtrado.rename(columns={"MIBGAS Daily Price [EUR/MWh]": "precioGas", "Trading day": "fecha"})

    # NO HACE FALTA EN VERSION FINAL - Convertimos la columna "fecha" al tipo datetime
    #df_aux["fecha"] = pd.to_datetime(df_aux["fecha"], dayfirst=True, utc=False)
    #cols=["fecha", "precioGas"]

    # Revertimos el orden de los datos para que este sea de mayor a menor antiguedad
    # df_aux = df_aux.sort_values(by="fecha", ascending=True)

    # Repetimos cada valor de la columna 'precioGas' 24 veces para simular precio por
    df_pg = df_aux.loc[df_aux.index.repeat(24)].reset_index(drop=True)

    cols=["precioGas"]
    
    return df_pg[cols]

def procesar_datos_temperatura(datos):
    """
    Procesa los datos sobre las temperaturas máximas y mínimas
    contenidas en un archivo json.
    ---------------------------------------------------------   
    Datos: ruta del archivo json que contiene los datos climatológicos.
    Devuelve: Dataframe pandas que contiene los datos de precio del gas
        después de cambiar el nombre de la columna a "precioGas".

    """
    df = pd.read_json(datos)
    #Extraemos nombre de la ciudad - particularizado para nuestro caso
    nombre = df["nombre"][0].capitalize()
    nombre = nombre.split(" ")[0]
    nombre = nombre.split(",")[0]
    df.rename(columns={"tmin":"tmin"+nombre, "tmax":"tmax"+nombre}, inplace=True)
    cols=["tmin"+nombre, "tmax"+nombre]
    df_temp = df[cols]
    df_temp = df_temp.loc[df_temp.index.repeat(24)].reset_index(drop=True)
    return df_temp

def genera_timestamps(start_date_str, end_date_str):
    """
    TEMPORAL HASTA SOLUCIONAR PROBLEMA DE DATOS FALTANTES
    Método auxiliar, enumera todas las fechas y horas comprendidas entre dos dadas,
    para comprobar qué fechas son las que faltan entre los datos descargados de esios
    ----------------------------------------------------------------------
    Entrada: str, fechas de inicio y final
    Salida: conjunto con todos los timestamps entre ambas fechas
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')
    
    current_date = start_date
    ts = set()

    while current_date <= end_date:
        ts.add(current_date)
        current_date += timedelta(hours=1)

    return ts

# Codificación cíclica de las variables de calendario y luz solar
# ==============================================================================
def codificacion_ciclica(datos: pd.Series, longitud_ciclo: int) -> pd.DataFrame:
    """
    Codifica una variable cíclica con dos nuevas variables: seno y coseno.
    Se asume que el valor mínimo de la variable es 0. El valor máximo de la
    variable se pasa como argumento.
    ----------
    Entrada:
   datos: pd.Series
        Serie con la variable a codificar.
    longitud_ciclo : int
        La longitud del ciclo. Por ejemplo, 12 para meses, 24 para horas, etc.
        Este valor se utiliza para calcular el ángulo del seno y coseno.

    Returns
    -------
    resultado : pd.DataFrame
        Dataframe con las dos nuevas características: seno y coseno.

    """

    seno = np.sin(2 * np.pi * datos/longitud_ciclo)
    coseno = np.cos(2 * np.pi * datos/longitud_ciclo)
    resultado =  pd.DataFrame({
                  f"{datos.name}_seno": seno,
                  f"{datos.name}_coseno": coseno
              })

    return resultado

def main():
    """
    Generacion de dataframe con todos los datos de interes 
    """
    ### TESTS DE CADA UNO DE LOS DATOS DE LECTURA DE FICHEROS
    #procesar_datos_demanda(datosDemanda)
    #procesar_datos_precio(datosPrecio)
    #procesar_datos_GeneracionSolar(datosGeneracionSolar)
    #procesar_datos_GeneracionEolica(datosGeneracionEolica)
    #procesar_datos_gas(datosGas)
    #procesar_datos_temperatura(datosTempValencia)
    df = obtener_tabla_datos(datosDemanda, datosPrecio, datosGeneracionSolar, datosGeneracionEolica,
                        datosGas, datosTempBarcelona, datosTempValencia, datosTempMadrid,
                        datosTempSevilla)
    
    '''
    ### Comprobacion de fechas ausentes
    inicio = "2022-01-01 0:0:0"
    fin = "2023-12-31 0:0:0"

    cf = set(genera_timestamps(inicio, fin))
    fechas = set(df["fecha"])
    print(cf-fechas)
    '''

if __name__ == "__main__":
    main()

