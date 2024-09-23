# El siguiente código a continuación está disponible en el repositorio de GitHub:
# https://github.com/SebaMorales74/machine-learning-inacap-evaluacion-1
# Se fabricó en colaboración con Sebastián Morales y Loreto Peñaloza.
# El proyecto se planificó para que se estructurara por módulos, cada uno con una función específica.
# Pero para una entrega más sencilla, se unieron los módulos en un solo archivo.
# Para empezar a ejecutar el programa, se debe de ejectuar este mismo Script.

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


"""
Módulo: modelo.py
=====

Paquetes requeridos
-------------------
    1. pandas
    2. scikit-learn
    3. matplotlib

Descripción
-----------
Este módulo se encarga de tomar los datos generados por el módulo generadorDeDatos y realizar una predicción del nivel de participación de los usuarios en la red social.
Se utiliza la clase Modelo para realizar la predicción del nivel de participación de los usuarios en la red social.

    >>> modelo = Modelo(datos) # -> Se crea una instancia de la clase Prediccion.
    >>> modelo.prediccionDelNivelDeParticipacion() # --> Realiza la predicción del nivel de participación de los usuarios en la red social.
"""

# Se importan los paquetes necesarios.
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import pandas
except ImportError as e:
    print(
        f'Error al importar los paquetes necesarios. Asegurate de tener instalados los paquetes: pandas, scikit-learn, matplotlib.\n\n{e}')


class Modelo:
    """
    Modelo
    ----------

    Modelo que aplica machine learning en distintos niveles y tipos.
    """

    def __init__(self, datos: pandas.DataFrame):
        """
        Constructor
        -----------
        Se inicializa la clase con un artributo privado para guardar en la instancia los datos recibidos en su inicialización.\n

        Parámetros
        ----------
        * datos : pandas.DataFrame = Datos generados por la clase **GeneradorDeDatos** o un DataF rame con al menos los campos:
            1. 'friends'
            2. 'postFrequency'
            3. 'averageComments'
            4. 'averageShares'
            5. 'averageLikes'.
        """
        self.__datos = datos

    def aprendizajeSupervisado(self):
        """
        Método aprendizajeSupervisado
        -------------------
        Aprendizaje supervisado: Regresión lineal.\n
        Prediccion del nivel de participación de los usuarios en la red social.\n

        Genera una cantidad de datos aleatorios definida por el parámetro "cantidad" de tipo entero.
        Para que los datos posean cierta corelación entre ellos, se implementa una fórmula para generar los likes.\n
        Los datos generados son guardados en el atributo privado "__datos" de tipo DataFrame de pandas.\n
        """
        # Se seleccionan las columnas que se utilizarán para la predicción.
        x = self.__datos[
            ['friends',
             'postFrequencyCode',
             'averageComments',
             'averageShares']
        ]

        # Se selecciona la columna que se desea predecir.
        y = self.__datos['averageLikes']

        # Se dividen los datos en entrenamiento y prueba.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)

        # Se crea el modelo de regresión lineal.
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Validación cruzada
        cv_scores = cross_val_score(model, x, y, cv=5, scoring='r2')

        # Resultados 📑
        print(
            f"""
            {"=" * 41}
            R² de validación cruzada: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})
            {"=" * 41}
            """
        )

        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Error cuadrático medio: {mse:.2f}")
        print(f"Coeficiente de determinación R²: {r2:.2f}")

        print(f"\nConfianza de modelo: {r2*100:.0f}%")
        if 0.8 <= r2:
            print("🟢 El modelo es aceptable 🎉")
        elif 0.5 <= r2 < 0.8:
            print("🟡 Se recomienda revisar posibles mejoras en el modelo.")
        else:
            print("🔴 El modelo no es aceptable. Revisar posibles mejoras.")

        print("\nCoeficientes del modelo:")
        for feature, coef in zip(x.columns, model.coef_):
            print(f"{feature}: {coef:.2f}")
        print(f"Intercepto: {model.intercept_:.2f}")

        # Gráfico de comparación entre likes reales y predichos.
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--',
            lw=2
        )
        plt.xlabel('Likes reales')
        plt.ylabel('Likes predichos')
        plt.title('Comparación entre likes reales y predichos')
        plt.show()

    def aprendizajeNoSupervisado(self):
        """
        Método aprendizajeNoSupervisado
        -------------------
        Aprendizaje no supervisado: Clustering.\n
        Prediccion del nivel de participación de los usuarios en la red social.\n

        Genera una cantidad de datos aleatorios definida por el parámetro "cantidad" de tipo entero.
        """
        datos = self.__datos.copy()

        # Seleccionar las características para clustering
        caracteristicas_clustering = ['friends', 'averageLikes']
        X_cluster = datos[caracteristicas_clustering].copy()

        # Mapear frecuencia a valores numéricos
        mapa_frecuencia = {'baja': 1, 'media': 2, 'alta': 3}
        X_cluster['postFrequency'] = datos['postFrequency'].map(
            mapa_frecuencia)

        # Escalar las características
        escalador = StandardScaler()
        X_escalado = escalador.fit_transform(X_cluster)

        # K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        datos['cluster'] = kmeans.fit_predict(X_escalado)

        plt.figure(figsize=(12, 8))
        dispersion = plt.scatter(X_cluster['friends'], X_cluster['averageLikes'],
                                 c=datos['cluster'], cmap='viridis')
        plt.xlabel('Número de amigos')
        plt.ylabel('Me gusta promedio por publicación')
        plt.title('Segmentación de usuarios')
        plt.colorbar(dispersion)
        plt.show()

        print("\nDistribución de usuarios por cluster:")
        print(datos['cluster'].value_counts())

        # Características promedio por cluster
        medias_cluster = datos.groupby('cluster')[
            caracteristicas_clustering + ['postFrequency']].mean(numeric_only=True)
        # Desescalar las características
        print("\nCaracterísticas promedio por cluster:")
        print(medias_cluster)


def main():
    gdd: GeneradorDeDatos = GeneradorDeDatos()
    gdd.generarDatos(100)
    datos = gdd.getDatos()

    modelo: Modelo = Modelo(datos)
    modelo.aprendizajeSupervisado()
    modelo.aprendizajeNoSupervisado()


if __name__ == '__main__':
    main()
