import pandas as pandas
import numpy as numpy

def initialize_discretization(attribute):
    # Paso 1.1: Encontrar el valor mínimo y máximo
    d0 = attribute.min()
    dn = attribute.max()

    # Paso 1.2: Formar un conjunto de todos los valores distintos de Fi, colocados en orden ascendente
    unique_values = numpy.unique(attribute)
    B = numpy.sort(numpy.append([d0, dn], unique_values))

    # Paso 1.3: Crear una variable D que contendrá el esquema de discretización
    D = [d0, dn]
    GlobalCaim = 0

    return B, D, GlobalCaim

def calculate_caim(D, attribute, classes):
    return 0

def update_discretization(B, D, GlobalCaim, attribute, classes, S):
    # Paso 2.1: Inicializar una variable K = 1
    K = 1

    while True:
        # Paso 2.2: Tentativamente agregar algunos límites, de B que no se encuentren ya en D y calcular su correspondiente valor CAIM
        potential_cut_points = [b for b in B if b not in D]
        caims = [calculate_caim(D + [b], attribute, classes) for b in potential_cut_points]

        # Paso 2.3: Aceptar el límite que posea el valor mayor de CAIM
        max_caim_index = numpy.argmax(caims)
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

def main():
    # Se cargan los datos
    data = pandas.read_csv('iris.data')
    attributes = data.iloc[:, 0:4].values
    classes = data.iloc[:, 4].values
    S = len(numpy.unique(classes))

    # Inicializar la discretización para cada atributo
    for attribute in attributes.T:
        B, D, GlobalCaim = initialize_discretization(attribute)
        D, GlobalCaim = update_discretization(B, D, GlobalCaim, attribute, classes, S)
        print (D)

if __name__ == "__main__":
    main()