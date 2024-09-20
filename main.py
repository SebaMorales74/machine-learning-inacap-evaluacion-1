from components.generadorDeDatos import *
from components.predicciones import *


def main() -> None:
    gdd: GeneradorDeDatos = GeneradorDeDatos()
    gdd.generarDatos(100)
    datos = gdd.getDatos()
    print(datos)


if __name__ == '__main__':
    main()
