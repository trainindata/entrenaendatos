---
layout: post
title:  "Código en Python de ingeniería de variables"
author: sole
categories: [feature engineering, machine learning ]
image: assets/images/computer.jpg
---

Según una encuesta de [Forbes](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/?sh=23cce0296f63), 
los científicos de datos dedican alrededor del 60% de su tiempo a preparar los datos para el análisis de datos y el aprendizaje 
automático. Una gran parte de ese tiempo se dedica a la ingeniería de variables.

La ingeniería de variables es el proceso de construir variables útiles para entrenar algoritmos de aprendizaje automático. 
Es un paso crucial en todo flujo de aprendizaje automático, y consume bastante tiempo.

La ingeniería de variables involucra procesos como la imputación de valores faltantes, la codificación de variables 
categóricas, la transformación y discretización de variables numéricas, la eliminación o censura de valores atípicos y el 
escalado de atributos, entre otros procesos.

En este artículo, voy a discutir implementaciones de Python de ingeniería de variables para aprendizaje automático.


## Serie: Ingeniería de variables para aprendizaje automático

Este artículo forma parte de una serie de artículos sobre ingeniería de variables para datos tabulares. Estás leyendo 
el artículo nº 8. Puedes encontrar otros artículos en los siguientes enlaces:

1.	[Ingeniería de variables para machine learning](https://www.entrenaendatos.com/ingenieria-de-variables-para-machine-learning/)
2.	Imputación de datos faltantes
3.	Codificación de variables categóricas
4.	[Transformaciones estabilizadoras de varianza](https://www.entrenaendatos.com/transformaciones-estabilizadoras-de-varianza/)
5.	Magnitud y escalamiento de las variables
6.	Discretización
7.	Creación de nuevos atributos
8.	**Implementación en Python de técnicas ingeniería de variables**

Si quieres saber más sobre estas y otras técnicas de ingeniería de variables, consulta nuestro curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

Empecemos.

Vamos a comparando las implementaciones de los paquetes de Python para ingeniería de variables más utilizados, que son:

* [Scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)
* [Feature-engine](https://feature-engine.readthedocs.io/en/latest/)
* [Category Encoders](http://contrib.scikit-learn.org/category_encoders/)

Muchas de las técnicas de ingeniería de variables aprenden parámetros a partir de los datos. Por ejemplo, para imputar 
datos con la media, derivamos la media del set de entrenamiento. Para codificar variables categóricas, definimos mapeos 
de cadenas de caracteres a números a partir de los datos de entrenamiento también. La transformación de Box-Cox también 
necesita aprender el exponente óptimo para transformar los datos del conjunto de entrenamiento.

Es por esto que los paquetes Python tienen la funcionalidad de aprender y almacenar primero estos parámetros, y luego 
utilizarlos para transformar los datos entrantes.

En este artículo, nos centraremos en Scikit-learn, Feature-engine y Category encoders, porque tienen precisamente esta 
funcionalidad. No hablaremos entonces, de la librería pandas, que, aunque contiene métodos intrínsecos para, por ejemplo, 
imputar los datos que faltan (`fillna()`), o mapear (`map()`) un valor de una variable a algún otro valor, no contiene 
la funcionalidad para aprender y perpetuar los parámetros necesarios.

## Paquetes de Python para ingeniería de variables

Los paquetes [Scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing), 
[Feature-engine](https://feature-engine.readthedocs.io/en/latest/) y 
[Category Encoders](http://contrib.scikit-learn.org/category_encoders/) tienen en común la funcionalidad fit() y transform() 
para aprender parámetros de los datos, y luego transformar las variables.

Asimismo, hay algunas diferencias tecnicas entre estos paquetes en términos de i) el resultado que derivan, ii) si operan 
en todo el set de datos o en una porción, y iii) si permiten la optimización de hyperparametros.

### Matriz de NumPy vs DataFrame de Pandas

La ingeniería de variables se realiza antes del entrenamiento de los modelos de aprendizaje automático. A menudo, queremos 
entender cómo estas transformaciones afectan a las variables y sus relaciones entre sí. Pandas es una gran herramienta 
para el análisis y la visualización de datos, y por lo tanto, las librerias que retornan dataframes de pandas en lugar de 
arrays de NumPy son más "amigables" para el análisis de datos.

Los paquetes Feature-engine y Category Encoders retornan dataframes de pandas, mientras que Scikit-learn retorna arrays 
de NumPy. Las matrices NumPy están optimizadas para el aprendizaje automático, ya que NumPy es computacionalmente más 
eficiente, pero no tienen functionalidad intrínseca para visualización de datos, como si la tienen los dataframes de pandas. 

### Transformación de algunas vs todas las variables

Las técnicas de ingeniería de variables suelen aplicarse a diferentes subconjuntos de variables. Por ejemplo, sólo imputamos 
las variables que contienen datos perdidos, y no necesariamente todo el conjunto de datos. Además, hay técnicas de imputación 
más adecuadas para las variables numéricas y otras más adecuadas para las variables categóricas.

Del mismo modo, podemos querer discretizar un grupo de variables mientras transformamos matemáticamente otro. Por lo tanto, 
la capacidad de seleccionar variables dentro del mismo transformador, nos proporciona mayor versatilidad y sencillez al 
momento de transformar variables.

Los paquetes Feature-engine y Category Encoders nos permiten seleccionar las variables a transformar desde el mismo 
transformador. Por otro lado, los transformadores de Scikit-learn operan sobre todo el conjunto de datos; lo que significa 
que tenemos que dividir el dataframe en los subgrupos de variables a los que aplicaremos cada técnica antes de usar los 
transformadores de Scikit-learn. Esto lo podemos hacer manualmente usando pandas, con la ayuda del `ColumnTransformer()` 
de Scikit-learn, o del `SklearnWrapper()` de Feature-engine. ¡Lo bueno de usar el `SklearnWrapper()`, es que la salida 
es un dataframe de pandas!

### Optimización de hyperparametros

A veces, podemos preguntarnos qué técnica de transformación produce la variable más predictiva. Por ejemplo, ¿debemos hacer 
una discretización en intervalos de igual distancia o de igual frecuencia? ¿Debemos imputar con la media, la mediana o un 
número arbitrario? ¿Debemos transformar con el logaritmo o quizá con alguna otra función matemática?

La mayoría de los transformadores de Scikit-learn son centralizados, es decir, que un transformador, puede realizar diferentes 
transformaciones. Por ejemplo, podemos aplicar 3 técnicas de discretización simplemente cambiando los parámetros de la clase 
`KBinsDiscretizer()` de Scikit-learn, mientras que, Feature-engine ofrece 3 transformadores diferentes para la discretización. 

Lo mismo ocurre con la imputación; cambiando los parámetros de `SimpleImputer()` de Scikit-learn, podemos realizar diferentes 
técnicas de imputación, mientras que Feature-engine ofrece varios transformadores, cada uno de los cuales puede realizar 
como máximo 2 técnicas de imputación diferentes.

Esto le da una versatilidad adicional a los transformadores de Scikit-learn al momento de hacer búsqueda en cuadrilla para 
la optimización de hyperaparametros. No es que con Feature-engine no se pueda hacer, pero se requieren más líneas de código.

En adelante, compararemos implementaciones de código para la imputación de datos faltantes, la codificación categórica, la 
transformación matemática y la discretización de variables ofrecidas por Scikit-learn, Feature-engine y los Category Encoders.

## Imputación de datos perdidos
 
![walking]({{ site.baseurl }}/assets/images/missingdata.jpg)

La imputación es el proceso de sustituir los datos que faltan en una variable, por un valor estimado de la información 
disponible en el conjunto de datos, normalmente dentro de la misma variable. Existen diferentes técnicas de imputación de 
datos faltantes, cada una de las cuales sirve para diferentes propósitos.

Si quieres saber más sobre estas técnicas, sus ventajas y limitaciones y cuándo debemos utilizarlas, consulta el curso 
[Ingeniería de variables para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

Scikit-learn y Feature-engine ofrecen una variedad de transformadores para la imputación de datos para variables numéricas 
y categóricas. Estas librerias tienen sutiles diferencias en la implementación y el resultado.

Como ya hemos dicho, Feature-engine retorna dataframes de pandas mientras que Scikit-learn retorna arrays de NumPy tras la 
imputación.

Ambos paquetes ofrecen las transformaciones mas comunes, que son con la media, la mediana, la moda y con un valor arbitrario. 

Feature-engine ofrece además, imputación con muestras aleatorias, con valores extremos, y análisis de datos completos. Por 
otro lado, Scikit-learn ofrece imputación multivariada.

Feature-engine nos permite seleccionar las variables que queremos imputar utilizando el mismo transformador, mientras que los 
transformadores de Scikit-learn imputarán el set de datos entero. Si queremos imputar solo algunas de las variables, debemos 
hacer uso de un transformador auxiliar, el `ColumnTransformer`.

Los transformadores del Feature-engine pueden identificar automáticamente las variables numéricas y categóricas, dependiendo 
de la técnica de imputación que queramos aplicar. De este modo, no acabaremos añadiendo inadvertidamente una cadena de 
caracteres cuando imputemos variables numéricas, o un número a las variables categóricas.

Por último, el mismo transformador de Scikit-learn, `SimpleImputer()`, puede realizar todas las técnicas de imputación 
simplemente ajustando los parámetros `strategy` y `fill_value`.

Para comparar la implementación de ambos paquetes, realizaremos imputación con la mediana y luego imputación con la categoría 
más frecuente.

Para la imputación con la mediana, que sólo es aplicable a variables numéricas, Feature-engine ofrece la clase 
`MeanMedianImputer()`, y Scikit-learn ofrece al `SimpleImputer()`.

`MeanMedianImputer()` de Feature-engine selecciona automáticamente todas las variables numéricas en el conjunto de datos 
de entrenamiento, dejando fuera las variables categóricas, mientras que `SimpleImputer()` de Scikit-learn transforma todas 
las variables en el conjunto de datos y genera un error si hay variables categóricas durante la ejecución.

El `SimpleImputer()` también ofrece la imputación de datos categóricos, utilizando la estrategia `'most_frequent'` o 
`'constant'`. Sin embargo, al utilizar cualquiera de estas estrategias de imputación, la transformación se aplica 
automáticamente tanto a las variables numéricas como a las categóricas, aunque están pensadas casi exclusivamente para 
ser utilizadas en variables categóricas. En esos casos, las variables numéricas cambiaran su formato a objetos, asi que 
hay que tener cuidado. 

Feature-engine tiene al transformador `CategoricalImputer()`, que selecciona automáticamente las variables categóricas 
para la imputación si no se declaran específicamente.

Veamos estos transformadores en acción.

### Imputación de la media/mediana

Para las demostraciones, utilizamos el conjunto de [datos de precios de la vivienda de Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

#### Feature-engine

A continuacion, podemos ver la implementación de `MeanMedianImputer()` utilizando la mediana como método de imputación. 
La imputación con la media puede implementarse de forma similar sustituyendo `"median"` por `"mean"` en el parámetro 
`imputation_method`.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer 

# Cargar los datos
data = pd.read_csv('houseprice.csv')

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
     test_size=0.3,
    random_state=0,
)

# configurar la imputación
median_imputer = MeanMedianImputer(
    imputation_method='median', 
    variables=['LotFrontage', 'MasVnrArea'],
) 

# ajustar la imputación
median_imputer.fit(X_train) 

# transformar los datos
train_t = median_imputer.transform(X_train)
test_t = median_imputer.transform(X_test)
```

Feature-engine retorna el set de datos original, en el que sólo se han modificado las variables deseadas. 

#### Scikit-learn

Utilizando el SimpleImputer() podemos hacer imputación con la media o la mediana definiendo el método con el parámetro 
`strategy`. Por ejemplo, para hacer imputación con la mediana:

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Cargar los datos
data = pd.read_csv('houseprice.csv') 

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
) 

# Configurar el imputador
median_imputer = SimpleImputer(strategy='median') 

# ajustar el imputador
median_imputer.fit(X_train[['LoteFronte', 'MasVnrArea']]) 

# transformar los datos
X_train_t = median_imputer.transform(X_train[['LoteFronte', 'MasVnrArea'])
X_test_t = median_imputer.transform( X_test[['LoteFronte', 'MasVnrArea']])
```

Como podemos ver arriba, Scikit-learn requiere que recortemos el de datos antes o mientras se lo pasamos a la función de 
imputación, mientras que este paso no era necesario con Feature-engine. El resultado es tambien un array NumPy con sólo 
las 2 variables numéricas imputadas.

### Imputación de categorías frecuentes

Este método se aplica a las variables categóricas y sustituye los datos que faltan por la categoría más frecuente (es 
decir, la moda), identificada en las variables del conjunto de entrenamiento. Los siguientes pasos muestran la imputación 
de la categoría más frecuente.

#### Feature-engine

El `CategoricalImputer()` sustituye los datos que faltan en las variables categóricas por la moda cuando establecemos el 
parámetro `imputation_method` como `'frequent'`. Se puede declarar una lista de variables, como se hace a continuación, 
de lo contrario, el transformador seleccionará automáticamente todas las variables categóricas del conjunto de datos de 
entrenamiento.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.imputation import CategoricalImputer

# Cargar los datos
data = pd.read_csv('houseprice.csv') 

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
)

# configurar la imputación
imputer = CategoricalImputer(
    imputation_method='frequent',
    variables=['Alley', 'MasVnrType'],
) 

# ajustar la imputación
imputer.fit(X_train) 

# transformar los datos
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)
```

Obtenemos asi, un set de datos completo, donde las 2 variables indicadas carecen de datos faltantes.

#### Scikit-learn

La clase `SimpleImputer()` también se utiliza para la imputación de categorías frecuentes utilizando `"most_frequent"` 
como estrategia de imputación. Sin embargo, las variables categóricas deben declararse explícitamente en este caso, ya 
que la estrategia de imputación se aplicará tanto a las variables numéricas como a las categóricas si se deja sin 
especificar.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Cargar dataset
data = pd.read_csv('houseprice.csv')
 
# Separar en conjuntos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
) 

# configurar el imputador
mode_imputer = SimpleImputer(strategy='most_frequent') 

# ajustar el imputador
mode_imputer.fit(X_train[['Alley', 'MasVnrType']]) 

# transformar los datos
X_train = mode_imputer.transform( X_train[['Callejón', 'MasVnrType']])
X_test = mode_imputer.transform( X_test[['Callejón', 'MasVnrType']])
```

La salida en este caso, `X_train` y  `X_test`, es un array Numpy con 2 columnas. En su lugar, podríamos asignar el 
resultado al set de datos original para reemplazar los valores de la columna original.

## Codificación de variables categóricas
 
Las variables categóricas tienen cadenas de caracteres en lugar de números. Los modelos de aprendizaje automático requieren 
datos en formato numérico, lo que hace necesario convertir a las variables categóricas en variables numéricas. Esto se hace 
mediante la codificación. El método de codificación que elijamos depende de los datos.

![walking]({{ site.baseurl }}/assets/images/etiquetas.png)


Scikit-learn, Feature-engine y Category encoders ofrecen una amplia gama de transformadores para variables categóricas. Los 
tres paquetes ofrecen los métodos de codificación más utilizados como codificación One-hot y Ordinal. 

Feature-engine y Category Encoders también ofrecen métodos de codificación basados en la variable objetivo, como la 
codificación con la media del objetivo o con del peso de la evidencia.

Feature-engine y Category encoders pueden detectar automáticamente las variables categóricas. Scikit-learn no. 

Feature-engine y Category encoders también nos permiten definir las variables que queremos codificar. Scikit-learn codificara 
el set de datos entero.

Category encoders es el paquete con mayor número de técnicas de codificación de variables categóricas. Las técnicas se 
basan en publicaciones científicas.

En los siguientes párrafos, compararemos la implementación de la codificación ordinal entre las 3 librerias de Python.

### Codificación ordinal

La codificación ordinal reemplaza los valores categóricos por números enteros arbitrarios. Por ejemplo, para una variable 
categórica con un número n de categorías únicas, la codificación ordinal sustituirá las categorías por dígitos de 0 a n-1.

#### Feature-engine

El `OrdinalEncoder()` de Feature-engine sólo funciona con variables categóricas. Se le puede indicar al transformador una 
lista de variables a transformar, o el codificador seleccionará automáticamente todas las variables categóricas en el 
conjunto de entrenamiento.

El `OrdinalEncoder()` sustituye las categorías por números, empezando por 0 hasta n-1, donde n es el número de categorías 
diferentes. Si seleccionamos `"arbitrary"` como método de codificación, el codificador asignará los números en la secuencia 
en que aparecen las etiquetas en la variable (es decir, por orden de llegada). Si se selecciona `"ordered"`, el codificador 
asignará los números siguiendo la media de la variable objetivo para esa etiqueta. A las etiquetas cuya media del objetivo 
sea mayor se les asignará el número 0, y a aquellas cuya media del objetivo sea menor se les asignará n-1.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OrdinalEncoder

# Obtener el set de datos
def load_titanic(): 
    data = pd.read_csv( 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl') 
    data = data.replace('?', np.nan) 
    data['cabin'] = data['cabin'].astype(str).str[0] 
    data['pclass'] = data['pclass']. astype('O') 
    data['embarked']. fillna('C', inplace=True) 
    return data 

data = load_titanic() 

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['survived', 'name', 'ticket'], axis=1), 
    data['survived'], 
    test_size=0.3, 
    random_state=0,
)
 
# configurar el codificador
encoder = OrdinalEncoder(
    encoding_method='arbitrary', 
    variables=['pclass', 'cabin', 'embarked'],
) 

# ajustar el codificador
encoder.fit(X_train, y_train) 

# transformar los datos
train_t = encoder.transform(X_train)
test_t = encoder.transform(X_test)
```

Feature-engine retorna un dataframe con las variables originales, y las variables categóricas ahora en formato numérico.

#### Scikit-learn

El `OrdinalEncoder()` de Scitkit-learn  transforma todas las variables en los datos de entrada. Por lo tanto, si queremos 
codificar solo un subconjunto, debemos cortar el set de datos o usar el `ColumnTransformer()`. Durante el proceso de 
codificación, los números se asignan simplemente por el orden alfabético de las etiquetas.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
def load_titanic(): 
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl') 
    data = data.replace('?', np.nan) 
    data['cabin'] = data['cabin'].astype(str).str[0] 
    data['pclass'] = data['pclass']. astype('O') 
    data['embarked']. fillna('C', inplace=True) 
    return data

data = load_titanic() 

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['survived', 'name', 'ticket'], axis=1), 
    data['survived'], 
    test_size=0.3, 
    random_state=0,
)

# configurar el codificador
encoder = OrdinalEncoder() 

# ajustar el codificador
encoder.fit(X_train[['pclass', 'cabin', 'embarked']], y_train)
 
# transformar los datos
train_t = encoder.transform( X_train[['pclass', 'cabin', 'embarked']])
test_t= encoder.transform( X_test[['pclass', 'cabin', 'embarked']])
```

La salida del bloque de código anterior es un array NumPy con (sólo) 3 columnas.

#### Category encoders

El `OrdinalEncoder()` de Category Encoders nos permite especificar las variables/columnas como parámetro. También se 
puede pasar un diccionario de mapeo opcional. Por defecto, se asume que las clases no tienen un orden real y los números 
se asignan a las etiquetas de forma aleatoria.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders.ordinal import OrdinalEncoder

# Load dataset
def load_titanic(): 
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl') 
    data = data.replace('?', np.nan) 
    data['cabin'] = data['cabin'].astype(str).str[0] 
    data['pclass'] = data['pclass']. astype('O') 
    data['embarked']. fillna('C', inplace=True) 
    return data

data = load_titanic() 

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['survived', 'name', 'ticket'], axis=1), 
    data['survived'], 
    test_size=0.3, 
    random_state=0,
)

# configurar el codificador
encoder = OrdinalEncoder(cols=['pclass', 'cabin', 'embarked']) 

# ajustar el codificador
encoder.fit(X_train, y_train) 

# transforma los datos
train_t = encoder.transform(X_train)
test_t = encoder.transform(X_test)
```

Y ahora tenemos un dataframe de pandas en donde las variables indicadas fueron codificadas con numeros.

## Transformación
 
![walking]({{ site.baseurl }}/assets/images/geometry.jpg)


Con el objetivo de obtener una distribución de aspecto más "gaussiano" o "normal" es comun transformar a las variables 
numéricas utilizando funciones matemáticas como por ejemplo, el logaritmo, la potencia y la función recíproca.

Scikit-learn ofrece la clase `FunctionTransformer()` que, en principio, puede aplicar cualquier función definida por el 
usuario. Toma la función como argumento, ya sea como un método NumPy, o como una función lambda.

Feature-engine, en cambio ofrece transformaciones matemáticas a través de transformadores específicos como el `LogTransformer()` 
y el `ReciprocalTransformer()`.

Para las transformaciones de Yeo-Johnson y Box-Cox, Scikit-learn centraliza las transformaciones dentro del `PowerTransformer()` 
(simplemente hay que cambiar el parametro `'method'`), mientras que Feature-engine tiene 2 transformadores individuales: 
el `YeoJohnsonTransformer` y el `BoxCoxTransformer`.

Ya hemos discutido las diferencias fundamentales entre ambos paquetes. Feature-engine genera un dataframe de pandas y 
selecciona automáticamente las variables numéricas o nos permite declarar las variables seleccionadas, mientras que 
Scikit-learn aplica la transformación a todo el dataframe y devuelve un array de NumPy.

Feature-engine devuelve un error si una transformación no es matemáticamente posible, por ejemplo log(0), o recíproco de 0, 
mientras que Scikit-learn introducirá NaNs en su lugar, necesitando que tu hagas una comprobación de racionalidad después.

En los próximos párrafos, compararemos la implementación de las transformaciones logarítmicas y Box-Cox entre los paquetes. 
Para las demostraciones, utilizamos el conjunto de [datos de precios de la vivienda de Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

### Transformación logarítmica

### Feature-engine

El `LogTransformer()` de Feature-engine aplica el logaritmo natural o el logaritmo de base 10 a variables numéricas. Sólo 
funciona con valores numéricos positivos. Si la variable contiene valores negativos, el transformador devolverá un error.

Como todos los transformadores de Feature-engine, éste también permite seleccionar las variables a transformar: Se puede 
pasar una lista de variables como argumento o, alternativamente, el transformador seleccionará y transformará 
automáticamente todas las variables numéricas.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.transformation import LogTransformer 

# Cargar el conjunto de datos
data = data = pd.read_csv('houseprice.csv')

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
) 

# configurar el transformador
tf = LogTransformer(variables = ['LotArea', 'GrLivArea']) 

# ajustar el transformador
tf.fit(X_train) 

# transformar los datos
train_t = tf.transform(X_train)
test_t = tf.transform(X_test)
```


#### Scikit-learn

Scikit-learn aplica la transformación logarítmica a través de su `FunctionTransformer()` pasando la función logarítmica 
como un método NumPy al transformador, como se muestra a continuación.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

# Cargar los datos
data = data = pd.read_csv('houseprice.csv')
 
# Separar en conjuntos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
)

# configurar el transformador
tf = FunctionTransformer(np.log) 

# ajustar el transformador
tf.fit(X_train[['LotArea', 'GrLivArea']]) 

# transformar los datos
train_t = tf.transform(X_train[['LotArea', 'GrLivArea']])
test_t = tf.transform(X_test[['LoteArea', 'GrLivArea']])
```


### Transformación de Box Cox

La transformación de Box Cox es una forma generalizada de transformaciones de potencia, en donde se evalúan varios exponentes, 
y se selecciona aquel que retorne una distribución de aspecto normal.

#### Feature-engine

El `BoxCoxTransformer()` aplica la transformación de Box Cox a las variables numéricas y sólo funciona con variables no 
negativas. Al igual que los demás transformadores de variables de Feature-engine, se puede pasar una lista de variables 
como argumento, o bien seleccionará y transformará automáticamente todas las variables numéricas.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.transformation import BoxCoxTransformer 

# Cargar el conjunto de datos
data = data = pd.read_csv('houseprice.csv')

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split( 
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
)

# configurar el transformador 
tf = BoxCoxTransformer(variables = ['LotArea', 'GrLivArea']) 

# ajustar el transformador
tf.fit(X_train) 

# transformar los datos
train_t = tf.transform(X_train)
test_t = tf.transform(X_test)
```

La transformación implementada por este transformador es la de `scipy.stats.boxcox` y el resultado es un dataframe de 
pandas.

#### Scikit-learn

Scikit-learn ofrece tanto la transformación de Box Cox como la de Yeo-Johnson a través de su `PowerTransformer()`. Box-Cox 
requiere que los datos de entrada sean valores estrictamente positivos.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

# Cargar el conjunto de datos
data = data = pd.read_csv('houseprice.csv') 

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split( 
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0,
) 

# configurar el transformador
tf = PowerTransformer(method="box-cox") 

# ajustar el transformador
tf.fit(X_train[['LotArea', 'GrLivArea']]) 

# transformar los datos
train_t = tf.transform(X_train[['LotArea', 'GrLivArea']])
test_t = tf.transform(X_test[['LoteArea', 'GrLivArea']])
```

El parámetro óptimo para estabilizar la varianza y minimizar la asimetría se estima mediante máxima verosimilitud. Como 
con todos los transformadores de Scikit-learn, los resultados se devuelven como un array de NumPy.

## Discretización
 
![walking]({{ site.baseurl }}/assets/images/discretizacion.png)

La discretización convierte, o particiona, las variables numéricas continuas en variables discretas de intervalos contiguos, 
que abarcan todo el rango de valores de la variable. La discretización suele aplicarse para mejorar la relación señal/ruido 
de una determinada variable y reducir los efectos de los valores atípicos.

Las diferencias en el tipo de salida y en los métodos de selección de variables entre los dos paquetes, tal y como hemos 
comentado anteriormente, siguen siendo válidas también para esta transformación.

Scikit-learn ofrece `KBinsDiscretizer()` como un transformador centralizado a través del cual podemos hacer discretización 
de igual distancia, igual frecuencia y k-means. Mientras que Feature-engine, ofrece transformadores individuales. Como el 
`EqualFrequencyDiscretiser()` y `EqualWidthDiscretiser()`.

Además, Scikit-learn nos permite seguir la discretización automáticamente con codificación one-hot de los intervalos. En 
el caso de Feature-engine, si deseamos tratar los bins como categorías, tendríamos que  hacerlo manualmente utilizando 
cualquiera de los codificadores categóricos a continuación de la discretización.

En los siguientes párrafos, compararemos la implementación de la discretización de igual frecuencia entre los paquetes.

### Discretización de igual frecuencia

Este tipo de discretización divide las variables en un número predefinido de intervalos contiguos. Los intervalos son 
normalmente definidos con los percentiles.

#### Feature-engine

El `EqualFrequencyDiscretiser()` particiona los valores de la variable numérica en intervalos contiguos de igual proporción 
de observaciones, donde los límites del intervalo se calculan según los percentiles.

Este número de intervalos en que debe dividirse la variable, lo determina el usuario. El transformador puede devolver la 
variable como numérica o como objeto (por defecto es numérica).

Inherente a Feature-engine, se puede indicar una lista de variables, o el discretizador seleccionará automáticamente todas 
las variables numéricas del conjunto de entrenamiento.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualFrequencyDiscretiser

# Cargar el conjunto de datos
data = data = pd.read_csv('houseprice.csv')

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3,
    random_state=0,
) 

# configurar la discretización 
disc = EqualFrequencyDiscretiser(q=10, variables=['LotArea', 'GrLivArea']) 

# ajustar el transformador
disc.fit(X_train) 

# transformar los datos
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)
```

El `EqualFrequencyDiscretiser()` primero encuentra los límites de los intervalos para cada variable, ya que se ajusta a 
los datos. Luego transforma las variables, ordenando los valores en los intervalos y devuelve un dataframe de pandas.

#### Scikit-learn

El paquete Scikit-learn puede implementar la discretización de frecuencias iguales a través de su transformador 
`KBinsDiscretizer()` ajustando el parámetro `"strategy"` a `"quantiles"`.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# Cargar el conjunto de datos
data = data = pd.read_csv('houseprice.csv')

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split( 
    data.drop(['Id', 'SalePrice'], axis=1), 
    data['SalePrice'], 
    test_size=0.3, 
    random_state=0, 
) 

# Configurar la discretización 
disc = KBinsDiscretizer(n_bins=10, strategy='quantile') 

# Ajustar el transformador
disc.fit(X_train[['LotArea', 'GrLivArea']]) 

# transformar los datos
train_t = disc.transform(X_train[['LotArea', 'GrLivArea']])
test_t = disc.transform(X_test[['LotArea', 'GrLivArea']])
```

Por defecto, el transformador retorna un array de NumPy dispersa. Podemos configurar métodos de codificación ordinal en 
su lugar, con el parámetro `"encode"`.

## Valores atípicos

Algunos algoritmos de aprendizaje automático son sensibles valores atípicos. Estos valores atípicos pueden ser el resultado 
de errores de medición/experimentales o de condiciones excepcionales del sistema y, por tanto, no tendrían valor predictivo. 
Por lo tanto, a veces ayuda censurar los valores de las variables a máximos o mínimos arbitrarios.

Feature-engine ofrece manejo de valores atípicos a través de Winsorizer() que limita o censura los valores máximos o mínimos 
de una variable a un valor arbitrario, y a través de OutlierTrimmer() elimina los valores atípicos del conjunto de datos.
 

## Para terminar...

Si quieres saber más sobre la ingeniería de variables consulta nuestro curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning), o el paquete 
de código abierto [Feature-engine](https://feature-engine.readthedocs.io/en/latest/) (ojo, la documentación está en inglés).

Para saber más sobre nuestros otros cursos, visita [Entrena en Datos](https://entrenaendatos.teachable.com/).

Espero que hayas encontrado el artículo útil, y si has llegado hasta aquí, gracias por leer.
