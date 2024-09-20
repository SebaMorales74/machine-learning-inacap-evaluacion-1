"""
Módulo: predicciones.py
=====

Paquetes requeridos
-------------------
    1. pandas
    2. scikit-learn
    3. matplotlib

Descripción
-----------
Este módulo se encarga de tomar los datos generados por el módulo generadorDeDatos y realizar una predicción del nivel de participación de los usuarios en la red social.
Se utiliza la clase Prediccion para realizar la predicción del nivel de participación de los usuarios en la red social.

    >>> pred = Prediccion(datos) # -> Se crea una instancia de la clase Prediccion.
    >>> pred.prediccionDelNivelDeParticipacion() # --> Realiza la predicción del nivel de participación de los usuarios en la red social.
"""

# Se importan los paquetes necesarios.
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from generadorDeDatos import *
    import pandas
except ImportError as e:
    print(
        f'Error al importar los paquetes necesarios. Asegurate de tener instalados los paquetes: pandas, scikit-learn, matplotlib.\n\n{e}')


class Prediccion:
    """
    Prediccion
    ----------

    Permite realizar una predicción del nivel de participación de los usuarios en la red social.
    """

    def __init__(self, datos: pandas.DataFrame):
        """
        Constructor
        -----------
        Se inicializa la clase con un artributo privado para guardar en la instancia los datos recibidos en su inicialización.\n

        Parámetros
        ----------
        * datos : pandas.DataFrame = Datos generados por la clase **GeneradorDeDatos** o un DataFrame con al menos los campos:
            1. 'friends'
            2. 'postFrequency'
            3. 'averageComments'
            4. 'averageShares'
            5. 'averageLikes'.
        """
        self.__datos = datos

    def prediccionDelNivelDeParticipacion(self):
        """
        Método prediccionDelNivelDeParticipacion
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
             'postFrequency',
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
        if  0.8 <= r2:
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


if __name__ == '__main__':
    print('🚧 TEST: Predicciones')
    gdd = GeneradorDeDatos()
    gdd.generarDatos(1000)
    datos = gdd.getDatos()

    prediccion = Prediccion(datos)
    prediccion.prediccionDelNivelDeParticipacion()
