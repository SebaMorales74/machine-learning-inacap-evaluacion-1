"""
Script principal.
Debes de ejecutar este archivo para que se ejecute el programa,
si deseas realizar pruebas, puedes hacerlo ejecutando los archivos de los módulos.
"""

from components.generadorDeDatos import *
from components.modelo import *


def main():
    gdd: GeneradorDeDatos = GeneradorDeDatos()
    gdd.generarDatos(100)
    datos = gdd.getDatos()

    modelo: Modelo = Modelo(datos)
    modelo.aprendizajeSupervisado()
    modelo.aprendizajeNoSupervisado()


if __name__ == '__main__':
    main()
