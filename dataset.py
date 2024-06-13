import pandas as pd


def data_preprocessing(data):
    df = data.drop(data.columns[0], axis=1)

    df['temp_media_Sevilla'] = (df['tmin_Sevilla'] + df['tmax_Sevilla']) / 2
    df['temp_media_Barcelona'] = (df['tmin_Barcelona'] + df['tmax_Barcelona']) / 2
    df['temp_media_Madrid'] = (df['tmin_Madrid'] + df['tmax_Madrid']) / 2
    df['temp_media_Valencia'] = (df['tmin_Valencia'] + df['tmax_Valencia']) / 2

    # Luego, calculamos la temperatura media en España
    df['temp_media_Ciudades'] = (
        df['temp_media_Sevilla'] +
        df['temp_media_Barcelona'] +
        df['temp_media_Madrid'] +
        df['temp_media_Valencia']
    ) / 4

    # Opcional: puedes eliminar las columnas intermedias si ya no las necesitas
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


class Dataset:
    def __init__(self, ruta="C:/Users/Jose/Desktop/TFG/data/datos_TFGb.xlsx"):
        data =   pd.read_excel(ruta, header=0)
        self.dataframe = data_preprocessing(data)
        self.column_ranges = self.calculate_column_ranges()
        self.max_per_type = [2,2]


    def calculate_column_ranges(self):
        column_ranges = {}
        for column in self.dataframe.columns:
            column_data = self.dataframe[column]
            column_type = str(column_data.dtype)  # Tipo de dato de la columna
            if column=='precio':
                column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible transactions':[2]}

            elif column_type == 'object':
                column_ranges[column] = {'type': 'Not quantitative', 'values': column_data.unique(), 'possible transactions': [0,1,2]}
            else:
                column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible transactions': [0,1,2]}
        return column_ranges
    


datos_prueb = Dataset()

    
    # print(DATAFRAME)
'''
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