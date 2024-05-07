import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt

# Función para inicializar la discretización
def initialize_discretization(attribute):
    # Paso 1.1: Encontrar el valor mínimo y máximo
    d0 = attribute.min()
    dn = attribute.max()

    # Paso 1.2: Formar un conjunto de todos los valores distintos de Fi, colocados en orden ascendente
    # Se obtienen los valores únicos de la columna
    unique_values = numpy.unique(attribute)
    # Se ordenan los valores únicos
    B = numpy.sort(numpy.append([d0, dn], unique_values))

    # Paso 1.3: Crear una variable D que contendrá el esquema de discretización
    D = [d0, dn]
    GlobalCaim = 0

    return B, D, GlobalCaim

# Función para calcular el CAIM, le mando el conjunto D que es la discretización, el atributo y las clases
def calculate_caim(D, attribute, classes):
    # Paso 3.1: Calcular el número de clases, de intervalos, de atributos y de instancias
    S = len(classes)
    M = len(attribute)
    N = len(D) - 1  # Número de intervalos

    # Crear la quanta matriz
    quanta_matrix = numpy.zeros((S, N))
    for i in range(N):
        # Calcular el número de instancias en el intervalo i
        for j, c in enumerate(classes):
            # Se obtiene el número de instancias en el intervalo i y en la clase c
            quanta_matrix[j, i] = ((D[i] <= attribute) & (attribute < D[i+1]) & (classes == c)).sum()

    # Calcular los totales de intervalo y de clase
    interval_totals = quanta_matrix.sum(axis=0)
    class_totals = quanta_matrix.sum(axis=1)

    # Calcular Maxr y M+r
    Maxr = quanta_matrix.max(axis=0)
    M_r = interval_totals

    # Calcular y devolver CAIM
    caim = ((Maxr / M_r) * (M_r / M)).sum()
    print(caim)
    return caim

# Función para actualizar la discretización, le mango el conjunto B que es el valor mínimo y máximo con los valores intermedios, D que es la discretización, GlobalCaim, el atributo, las clases y s que contiene la longitud de las clases
def update_discretization(B, D, GlobalCaim, attribute, classes, S):
    # Paso 2.1: Inicializar una variable K = 1
    K = 1

    while True:
        # Paso 2.2: Tentativamente agregar algunos límites, de B que no se encuentren ya en D y calcular su correspondiente valor CAIM
        potential_cut_points = [b for b in B if b not in D]
        caims = [calculate_caim(D + [b], attribute, classes) for b in potential_cut_points]

        # Paso 2.3: Aceptar el límite que posea el valor mayor de CAIM
        # Se obtiene el índice del valor máximo de CAIM
        max_caim_index = numpy.argmax(caims)
        # Se obtiene el valor máximo de CAIM y el mejor punto de corte
        max_caim = caims[max_caim_index]
        best_cut_point = potential_cut_points[max_caim_index]

        # Paso 2.4: Si (CAIM > GlobalCAIM or K < S) entonces actualiza D con el limite aceptado en el paso 2.3 y establece GlobalCAIM = CAIM, sino terminar el proceso
        if max_caim > GlobalCaim or K < S:
            D.append(best_cut_point)
            D.sort()
            GlobalCaim = max_caim
        else:
            break

        # Paso 2.5: establece K = K + 1
        K += 1

    return D, GlobalCaim

def plot_intervals(D, attribute, classes, subplot):
    # Crear un gráfico de dispersión de los datos en el subplot especificado
    unique_classes = numpy.unique(classes)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, c in enumerate(unique_classes):
        subplot.scatter(attribute[classes == c], [0]*len(attribute[classes == c]), marker='o', color=colors[i % len(colors)], label=c)

    # Dibujar y rellenar los intervalos
    interval_colors = ['#0000FF', '#008000', '#FF0000', '#00FFFF', '#FF00FF', '#FFFF00', '#000000', '#800000', '#808000', '#008080', '#800080', '#A52A2A', '#00008B', '#008B8B', '#B8860B']
    for i in range(len(D) - 1):
        subplot.axvspan(D[i], D[i+1], facecolor=interval_colors[i % len(interval_colors)], alpha=0.3)  # Reduced alpha to 0.3
        subplot.axvline(x=D[i], color='k')
        subplot.axvline(x=D[i+1], color='k')
        subplot.plot([], [], color=interval_colors[i % len(interval_colors)], alpha=0.3, linewidth=10, label=f'Intervalo {i+1}: [{D[i]}, {D[i+1]}]') 

    # Configurar el gráfico
    subplot.set_title('Data points and intervals')
    subplot.set_xlabel('Centímetros ')
    subplot.set_yticks([])
    subplot.legend()

def main():
    # Se cargan los datos
    data = pandas.read_csv('iris.data', header=None)
    #Se obtienen los atributos y las clases
    attributes = data.iloc[:, 0:4].values
    classes = data.iloc[:, 4].values
    #Se obtiene el número de clases
    S = len(numpy.unique(classes))
    #Se definen los nombres de los atributos
    attribute_names = ['longitud del sepalo', 'ancho del sepalo', 'longitud del petalo', 'ancho del sepalo']

    # Crear una figura para los subplots
    fig, axs = plt.subplots(2, 2)

   # Inicializar la discretización para cada atributo
    for i, attribute in enumerate(attributes.T):
        # Inicializar la discretización y asigno valores a B que contiene el valor mínimo, todos los valores intermedios y el valor máximo, D que es un array con la posición 0 que tiene el valor mínimo y la posición 1 con el valor máximo y GlobalCaim inicializado en 0
        B, D, GlobalCaim = initialize_discretization(attribute)
        # Actualizar la discretización
        D, GlobalCaim = update_discretization(B, D, GlobalCaim, attribute, classes, S)
        # Imprimir la discretización
        print (D)
        plot_intervals(D, attribute, classes, axs[i//2, i%2])
        axs[i//2, i%2].set_title(attribute_names[i])

    # Ajustar el espacio entre los subplots y mostrar la figura
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()