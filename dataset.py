import pandas as pd

class Dataset:
    data =   pd.read_excel("C:/Users/Jose/Desktop/TFG/data/datos_TFG.xlsx", header=0)
    df = data.drop(data.columns[0], axis=1)
    DATAFRAME = df.astype(float)

    def __init__(self):
        self.column_ranges = self.calculate_column_ranges()
        self.max_per_type = [2,2]


    def calculate_column_ranges(self):
        column_ranges = {}
        for column in self.DATAFRAME.columns:
            column_data = self.DATAFRAME[column]
            column_type = str(column_data.dtype)  # Tipo de dato de la columna
            if column=='precio':
                column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible transactions': [0,2]}

            elif column_type == 'object':
                column_ranges[column] = {'type': 'Not quantitative', 'values': column_data.unique(), 'possible transactions': [0,1,2]}
            else:
                column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible transactions': [0,1,2]}
        return column_ranges
    
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