import pandas as pandas
import numpy as numpy

def calculate_caim(cut_points, attribute, classes):
    None

def find_cut_points(attribute, classes, cut_points):
    max_caim = -np.inf
    best_cut_point = None
    for i in range(1, len(cut_points)):
        potential_cut_point = (cut_points[i-1] + cut_points[i]) / 2
        caim = calculate_caim(cut_points + [potential_cut_point], attribute, classes)
        if caim > max_caim:
            max_caim = caim
            best_cut_point = potential_cut_point
    return best_cut_point

def main():
    # Se cargan los datos
    data = pandas.read_csv('iris.data')
    attributes = data.iloc[:, 0:4].values
    classes = data.iloc[:, 4].values

    for attribute in attributes:
        cut_points = [attribute.min(), attribute.max()]  # Se inicializan los puntos de corte
        while True:
            cut_point = find_cut_points(attribute, classes, cut_points)
            if cut_point is None:
                break
            cut_points.append(cut_point)
            cut_points.sort()

if __name__ == "__main__":
    main()
