"""
Módulo: generadorDeDatos.py
=====

Paquetes requeridos
-------------------
    1. numpy
    2. pandas

Descripción
-----------
Este módulo se encarga de generar datos aleatorios para simular una base de datos de usuarios de una red social.
Se generan datos como: id, cantidad de amigos, frecuencia de publicaciones, categoría favorita, promedio de likes, comentarios y compartidos.\n
Se utiliza la clase GeneradorDeDatos para generar y guardar los datos en un atributo privado y se puede obtener los datos generados con el método getDatos()::
    
    >>> gdd = GeneradorDeDatos() # -> Se crea una instancia de la clase GeneradorDeDatos.
    >>> gdd.generarDatos(100) # --> Genera 100 datos aleatorios.
    >>> gdd.getDatos() # --> Retorna un DataFrame de pandas con los datos generados.
"""

# Se importan los paquetes necesarios.
try:
    import numpy
    import pandas
except ImportError as e:
    print(
        f'Error al importar los paquetes necesarios. Asegurate de tener instalados los paquetes: numpy, pandas.\n\n{e}')


class GeneradorDeDatos:
    """
    GeneradorDeDatos
    ----------------
    Permite generar datos aleatorios para simular una base de datos de usuarios de una red social.
    """

    def __init__(self):
        """
        Constructor
        -----------
        Se inicializa la clase con un artributo privado para guardar un DataFrame vacío.\n
        Dispone de los metodos `generarDatos()` y `getDatos()`.
        """
        self.__datos = pandas.DataFrame()  # Se inicializa un DataFrame de pandas vacío.

    def generarDatos(self, cantidad: int) -> None:
        """
        Método generarDatos
        -------------------
        Genera una cantidad de datos aleatorios definida por el parámetro "cantidad" de tipo entero.
        Para que los datos posean cierta corelación entre ellos, se implementa una fórmula para generar los likes.\n
        Los datos generados son guardados en el atributo privado "__datos" de tipo DataFrame de pandas.\n

        Parámetros
        ----------
        * cantidad : int = Cantidad de datos a generar.
        """
        print(f'Generando {cantidad} datos...')

        mapa_frecuencia = {'baja': 1, 'media': 2, 'alta': 3}

        # Se fija la semilla para obtener los mismos resultados en cada ejecución.
        numpy.random.seed(69)
        friends = numpy.random.randint(0, 600, size=cantidad)
        postFrequency = numpy.random.choice(
            ['baja', 'media', 'alta'], cantidad, p=[0.3, 0.5, 0.2])

        # Generamos likes con una relación más clara con friends y postFrequency.
        averageLikes = 5 + 0.1 * friends + 2 * numpy.vectorize(mapa_frecuencia.get)(postFrequency) + \
            numpy.random.normal(0, 10, size=cantidad)
        # Aseguramos que no haya likes negativos
        averageLikes = numpy.clip(averageLikes, 0, None)

        # Se crea un DataFrame de pandas con los datos generados.
        datos = pandas.DataFrame({
            'id': range(1, cantidad + 1),
            'friends': friends,
            'postFrequency': postFrequency,
            'postFrequencyCode': numpy.vectorize(mapa_frecuencia.get)(postFrequency),
            'favoriteCategory': numpy.random.choice(['Technology', 'Fashion', 'Food', 'Travel', 'Sports', 'Music', 'Photography', 'Art', 'Fitness', 'Pets'], size=cantidad),
            'averageLikes': averageLikes,
            'averageComments': numpy.random.randint(0, 50, size=cantidad),
            'averageShares': numpy.random.randint(0, 50, size=cantidad)
        })

        datos['frecuencia_numerica'] = datos['postFrequency'].map(
            mapa_frecuencia)
        datos['averageLikes'] = datos.apply(lambda fila: self.ajustarDatos(
            fila['friends'], fila['frecuencia_numerica'], fila['averageComments'], fila['averageShares'], fila['averageLikes'], 500), axis=1)
        datos['averageComments'] = datos.apply(lambda fila: self.ajustarDatos(
            fila['friends'], fila['frecuencia_numerica'], fila['averageComments'], fila['averageShares'], fila['averageComments'], 50), axis=1)
        datos['averageShares'] = datos.apply(lambda fila: self.ajustarDatos(
            fila['friends'], fila['frecuencia_numerica'], fila['averageComments'], fila['averageShares'], fila['averageShares'], 100), axis=1)

        datos = datos.drop('frecuencia_numerica', axis=1)

        print(f'Datos generados ✔\n\nCantidad:\n{datos.count()}\n')
        self.__datos = datos

    def ajustarDatos(self, friends, postFrequency, averageComments, averageShares, averageLikes, maxValue):
        """
        Método ajustarDatos
        -------------------
        Ajusta los datos. Se ajustan los likes, comentarios y compartidos en función de la cantidad de amigos, la frecuencia de publicaciones y los likes promedio.\n
        Se ajustan los datos para que no superen un valor máximo definido por el parámetro "maxValue" de tipo entero.\n
        """
        friends_factor = min(friends / 500, 2)
        postFrequency_factor = postFrequency / 2
        adjustment = int((averageLikes+averageComments+averageShares)
                         * friends_factor + postFrequency_factor)
        return min(adjustment, maxValue)

    def getDatos(self) -> pandas.DataFrame:
        """
        Método getDatos
        ------
        Obtiene los datos guardados en el atributo "datos".\n
        Retorna un DataFrame de pandas.
        """
        return self.__datos


# En caso de ejecutar este archivo como script, se generan datos de prueba e imprime en consola.
if __name__ == '__main__':
    print('🚧 TEST: Generador de datos')
    # Se crea una instancia de la clase GeneradorDeDatos.
    gdd = GeneradorDeDatos()
    gdd.generarDatos(100)  # Se generan 100 datos aleatorios.
    # Se imprimen los datos generados.
    print('Datos generados:\n', gdd.getDatos(), '\n')
