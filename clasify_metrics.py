import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def main():

    #df = pd.read_csv("C:/Users/Jose/Desktop/TFG/out/clasificacion_chisq.txt")


    # Leer el archivo .txt
    filename = "C:/Users/Jose/Desktop/TFG/out/clasificacion_confianza.txt"
    df = pd.read_csv(filename, header=0, delimiter=';')  # Cambiar el delimitador según el formato del archivo

    # Eliminar las columnas 'ID' y 'Fitness'
    df = df.drop(columns=['ID', 'Fitness'])

    # # Convertir todas las columnas a tipo numérico forzando errores a NaN (da errores si no luego con np.isinf)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Reemplazar valores infinitos y NaN en columnas para que la normalización dé unos
    df = df.replace([np.inf, -np.inf], np.inf).fillna(1)



    # Reemplazar valores infinitos en columnas para que la normalización dé unos
    for column in df.columns:
        if np.isinf(df[column]).any() or np.isneginf(df[column]).any():
            # Reemplazar todos los valores de la columna por 1
            df[column] = 1

    # Normalizar las columnas
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    normalized_df = normalized_df.drop(columns=['Lift', 'Leverage','Conviccion','Ganancia','Chi-sq'])

    # Asegurarse de que las columnas con infinitos sean todas unos después de la normalización
    for column in df.columns:
        if df[column].eq(1).all():
            normalized_df[column] = 1

    # Calcular la media por columna
    mean_per_column = normalized_df.mean()

    # print("DataFrame normalizado:")
    # print(normalized_df)
    print("\nMedia por columna:")
    print(mean_per_column)

        # Graficar los valores medios en un gráfico de barras
    plt.figure(figsize=(10, 6))
    mean_per_column.plot(kind='bar', color='lightblue')
    plt.title('Media por métrica (normalizada)')
    plt.xlabel('Métricas')
    plt.ylabel('Valor medio normalizado')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    plt.show()
        

if __name__ == "__main__":
    main()