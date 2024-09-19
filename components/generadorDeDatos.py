"""
Módulo: generadorDeDatos.py
=====

Paquetes requeridos
-------------------
    1. numpy
    2. pandas

Descripción:
------------
Este módulo se encarga de generar datos aleatorios para simular una base de datos de usuarios de una red social.
Se generan datos como: id, cantidad de amigos, frecuencia de publicaciones, categoría favorita, promedio de likes, comentarios y compartidos.\n
Se utiliza la clase GeneradorDeDatos para generar y guardar los datos en un atributo privado y se puede obtener los datos generados con el método getDatos()::

    >>> GeneradorDeDatos().generarDatos(100) # --> Genera 100 datos aleatorios.
    >>> GeneradorDeDatos().getDatos() # --> Retorna un DataFrame de pandas con los datos generados.
"""

# Se importan los paquetes necesarios.
import numpy
import pandas

# Clase GeneradorDeDatos


class GeneradorDeDatos:
    # Constructor. Se inicializa con un artributo privado para guardar una lista de datos.
    def __init__(self):
        self.__datos = []

    def generarDatos(self, cantidad: int) -> None:
        """
        Método. Genera datos aleatorios.
        """
        print(f'Generando {cantidad} datos...')
        numpy.random.seed(69)
        datos = pandas.DataFrame({
            'id': range(1, cantidad + 1),
            'friends': numpy.random.randint(0, 100, size=cantidad),
            'postFrequency': numpy.random.randint(1, 30, size=cantidad),
            'favoriteCategory': numpy.random.choice(['Technology', 'Fashion', 'Food', 'Travel', 'Sports', 'Music', 'Photography', 'Art', 'Fitness', 'Pets'], size=cantidad),
            'averageLikes': numpy.random.randint(0, 200, size=cantidad),
            'averageComments': numpy.random.randint(0, 50, size=cantidad),
            'averageShares': numpy.random.randint(0, 50, size=cantidad)
        })
        print(f'Datos generados ✔\n\nCantidad:\n{datos.count()}\n')
        self.__datos = datos

    def getDatos(self) -> pandas.DataFrame:
        """
        Método. Obtiene los datos guardados en el atributo "datos". Retorna un DataFrame de pandas.
        """
        return self.__datos


# En caso de ejecutar este archivo como script, se generan datos de prueba e imprime en consola.
if __name__ == '__main__':
    print('🚧 TEST: Generador de datos')
    gdd = GeneradorDeDatos()
    gdd.generarDatos(100)
    print('Datos generados:\n', gdd.getDatos(), '\n')
