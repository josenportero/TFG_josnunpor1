import pandas as pd

class Dataset:
    def __init__(self, dataframe):
        self.dataset = dataframe
        self.column_ranges = self.calculate_column_ranges()
        self.max_per_type = [2,2]


    def calculate_column_ranges(self):
        column_ranges = {}
        for column in self.dataset.columns:
            column_data = self.dataset[column]
            column_type = str(column_data.dtype)  # Tipo de dato de la columna
            if column_type == 'object':
                column_ranges[column] = {'type': 'Not quantitative', 'values': column_data.unique(), 'possible transactions': [0,1,2]}
            else:
                column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible transactions': [0,1,2]}
        return column_ranges
    
'''
# Ejemplo de uso
data = {
    'A': [1, 2, 3, 4, 5],
    'B': ['foo', 'bar', 'foo', 'bar', 'baz'],
    'C': [10.0, 20.5, 30.3, 40.8, 50.2]
}
df = pd.DataFrame(data)
dataset = Dataset(df)

# Imprimir dataset y los rangos de cada columna
print("Dataset:")
print(dataset.dataset)
print("\nRangos de columnas:")
print(dataset.column_ranges)

ls = []

for c in dataset.column_ranges:
    if dataset.column_ranges[c]['type']=='Quantitative':
        ls.append(dataset.column_ranges[c]['min'])
        ls.append(dataset.column_ranges[c]['max'])
print(ls)
'''