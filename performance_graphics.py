import matplotlib.pyplot as plt


def escalado_poblacion():
    # Datos para las pruebas de rendimiento
    individuos = [250, 500, 750, 1000]
    tiempo_mono = [9.731989860534668, 20.516558170318604, 32.92682719230652, 41.20526933670044]
    tiempo_multi = [40.179014444351196, 81.98537993431091, 121.70838665962219, 157.8533685207367]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))

    # Curva para el algoritmo mono-objetivo
    plt.plot(individuos, tiempo_mono,  linestyle='-', color='blue', label='Mono-objetivo')

    # Curva para el algoritmo multi-objetivo
    plt.plot(individuos, tiempo_multi,  linestyle='-', color='red', label='Multiobjetivo')

    plt.xlabel('Tamaño de la población')
    plt.ylabel('Tiempo Medio (segundos)')
    plt.title('Tiempo Medio de Ejecución vs Número de Individuos (Mono-objetivo vs Multiobjetivo)')
    plt.legend()
    plt.grid(True)
    plt.show()


def escalado_datos():
    # Datos para las pruebas de rendimiento
    instancias = [4343, 8759, 13102, 17518]
    tiempo_mono = [13.00, 14.01, 37.45, 42.89]
    tiempo_multi = [44.41, 68.67, 123.28, 158.67]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))

    # Curva para el algoritmo mono-objetivo
    plt.plot(instancias, tiempo_mono, marker='o', linestyle='-', color='blue', label='Mono-objetivo')

    # Curva para el algoritmo multi-objetivo
    plt.plot(instancias, tiempo_multi, marker='o', linestyle='-', color='red', label='Multiobjetivo')

    plt.xlabel('Número de Instancias')
    plt.ylabel('Tiempo Medio (segundos)')
    plt.title('Tiempo Medio de Ejecución vs Número de Instancias (Mono-objetivo vs Multiobjetivo)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    escalado_datos()


if __name__=='__main__':
    main()