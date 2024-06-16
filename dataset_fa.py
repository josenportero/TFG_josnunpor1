import pandas as pd


data =   pd.read_csv("C:/Users/Jose/Desktop/TFG/data/FA(in).csv", header=0,sep=';' )


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

    df.drop(['fecha'],axis=1, inplace=True)

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
            column_ranges[column] = {'type': 'Quantitative', 'min': column_data.min(), 'max': column_data.max(), 'possible types': [0,1,2]}
    return column_ranges

class Dataset:

    # Dataframe estático, no varía a lo largo de nuestro problema
    #dataframe = data_preprocessing(data)
    dataframe = data
    column_ranges = calculate_column_ranges(data)
    
