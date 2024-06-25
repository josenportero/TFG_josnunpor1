import pandas as pd
import matplotlib.pyplot as plt

data =   pd.read_excel("C:/Users/Jose/Desktop/TFG/data/datos_TFGb.xlsx", header=0)
AÑO = 0. # Para tests de rendimiento

def data_preprocessing(data):
    df = data.drop(data.columns[0], axis=1)

    # Creamos nuevas columnas 'año', 'mes' y 'dia'
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day

    # Quitamos la columna 'fecha' original si no es necesaria
    df = df.drop('fecha', axis=1)

    if AÑO!=0.:
        # Para tests los registros del año 2022
        df = df[(df['año'] == 2022) | ((df['año'] == 2023) & (df['mes'] < 7))] #
        #df = df[df['mes'] < 7]

    df['temp_media_Sevilla'] = (df['tmin_Sevilla'] + df['tmax_Sevilla']) / 2
    df['temp_media_Barcelona'] = (df['tmin_Barcelona'] + df['tmax_Barcelona']) / 2
    df['temp_media_Madrid'] = (df['tmin_Madrid'] + df['tmax_Madrid']) / 2
    df['temp_media_Valencia'] = (df['tmin_Valencia'] + df['tmax_Valencia']) / 2

    # Calculamos la temperatura media 'aprox' en España
    df['temp_media_Ciudades'] = (
        df['temp_media_Sevilla'] +
        df['temp_media_Barcelona'] +
        df['temp_media_Madrid'] +
        df['temp_media_Valencia']
    ) / 4

    # Eliminación de las columnas intermedias
    df.drop([
        'tmin_Sevilla', 'tmax_Sevilla',
        'tmin_Barcelona', 'tmax_Barcelona',
        'tmin_Madrid', 'tmax_Madrid',
        'tmin_Valencia', 'tmax_Valencia',
        'temp_media_Sevilla', 'temp_media_Barcelona',
        'temp_media_Madrid', 'temp_media_Valencia'
    ], axis=1, inplace=True)

    
    return df

def minmax_norm(df):
    for c in df.columns:
        df[c]=(df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df

def max_norm(df):
    for c in df.columns:
        df[c]=(df[c]) / (df[c].max())
    return df


def calculate_column_ranges(dataframe):
    column_ranges = {}
    for column in dataframe.columns:
        column_data = dataframe[column]
        column_type = str(column_data.dtype)  # Tipo de dato de la columna
        if column=='precio':
            column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible types':[2]}

        elif column_type == 'object':
            column_ranges[column] = {'type': 'Not quantitative', 'values': column_data.unique(), 'possible types': [0,1]}
        else:
            column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible types': [0,1]}
    return column_ranges

class Dataset:

    # Dataframe estático, no varía a lo largo de nuestro problema
    dataframe = data_preprocessing(data)
    column_ranges = calculate_column_ranges(dataframe)



# Histograma valores precio
# Dataset.dataframe['precio'].hist(bins=150)
# plt.xlabel('Valor')
# plt.ylabel('Frecuencia')
# plt.title('Histograma de precio')
# plt.show()

#print(Dataset.column_ranges['precio'])
#print(Dataset.dataframe)
'''
datos_prueb = Dataset()

    
    # print(DATAFRAME)

# Ejemplo de uso
data = {
    'A': [1, 2, 3, 4, 5],
    'B': ['foo', 'bar', 'foo', 'bar', 'baz'],
    'precio': [10.0, 20.5, 30.3, 40.8, 50.2]
}
df = pd.DataFrame(data)
for col in df.columns:
    print(col=='precio')

dataset = Dataset(df)

# Imprimir dataset y los rangos de cada columna
print("Dataset:")
print(dataset.dataframe)
print("\nRangos de columnas:")
print(dataset.column_ranges)

aux = [1,0,2]

def funaux(ls, dataset):
    ant = [i for i in  range(len(dataset.columns)) if ls[i]==1]
    cons = [i for i in range(len(dataset.columns)) if ls[i]==2]
    
    a_names = [df.columns[e] for e in ant]
    c_names = [df.columns[e] for e in cons]

    return f"IF {a_names} => THEN {c_names} "

res = funaux(aux, dataset.dataframe)

print(res)

ls = []

# for c in dataset.column_ranges:
#     if dataset.column_ranges[c]['type']=='Quantitative':
#         ls.append(dataset.column_ranges[c]['min'])
#         ls.append(dataset.column_ranges[c]['max'])
# print(ls)
'''