---
layout: post
title:  "Ingeniería de variables para machine learning"
author: sole
categories: [ feature engineering, machine learning ]
image: assets/images/5.png
---

La ingeniería de variables es el proceso de utilizar el conocimiento de los datos para transformar las variables 
existentes o crear nuevas variables, para su uso en modelos de aprendizaje automático.

Los datos en su formato original casi nunca son adecuados para entrenar algoritmos de aprendizaje automático. Los 
científicos de datos dedicamos una cantidad sustancial de tiempo al procesamiento de las variables para utilizarlas en 
modelos aprendizaje automático.

En este artículo, discutiremos los principales métodos de ingeniería de variables para abordar diversos aspectos de los 
atributos en nuestros datos.

## Serie: Ingeniería de variables para aprendizaje automático

Este artículo es parte de una serie de artículos sobre ingeniería de variables para datos tabulares. Estás leyendo el 
artículo nº 1. Puedes encontrar otros artículos en los siguientes enlaces:

1.	**Ingeniería de variables para machine learning**
2.	Imputación de datos faltantes
3.	Codificación de variables categóricas
4.	[Transformaciones estabilizadoras de varianza](https://www.entrenaendatos.com/transformaciones-estabilizadoras-de-varianza/)
5.	Magnitud y escalamiento de las variables
6.	Discretización
7.	Creación de nuevos atributos
8.	[Código en Python de ingeniería de variables](https://www.entrenaendatos.com/implementacion-en-Python-tecnicas-ingenieria-de-variables/)

Si quieres saber más sobre estas y otras técnicas de ingeniería de variables, consulta nuestro curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

Empecemos.

### ¿Por qué necesitamos transformar variables?

Hay varias razones por las que transformamos las variables en nuestros datos:

1.	Algunas librerias de aprendizaje automático, como Scikit-learn, no aceptan valores faltantes o cadenas de caracteres.
2.	Algunos modelos de aprendizaje automático son sensibles a la magnitud de las variables, por ejemplo, los modelos lineales, las máquinas de soporte vectorial y las redes neuronales y todos los algoritmos basados en distancias, como el PCA y los vecinos cercanos.
3.	Algunos algoritmos son sensibles a los valores atípicos, por ejemplo los modelos lineales.
4.	Algunas variables no son utilizables en su formato original, por ejemplo las fechas y horas.
5.	El procesamiento de las variables nos permite extraer más información, lo que puede aumentar el rendimiento del algoritmo, por ejemplo, la codificación variables categóricas con el peso de la evidencia.
6.	La combinación de variables puede generar atributos más predictivos que las variables originales, por ejemplo la suma o la media de un grupo de variables.

Como puedes ver, hay varias razones por las que nos gustaría modificar las variables en los datos. La ingeniería de 
variables se refiere a las diversas transformaciones que aplicamos para poder utilizar estas variables en los modelos de 
aprendizaje automático.

Ingeniería de variables es un término general que incluye múltiples métodos para realizar transformaciones, incluyendo 
reemplazar valores que faltan, codificar variables categóricas, transformar las variables con funciones y la crear nuevas 
variables.

En este artículo, presentatmos múltiples técnicas de ingeniería de variables para procesar los datos y dejarlos listos 
para su uso en modelos de aprendizaje automático. Describimos cada técnica y mencionamos brevemente cuándo debemos utilizarlas.

Puedes ver código y videos con tutoriales paso a paso en nuestro curso online [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

### Índice de contenidos

1.	Imputación de datos faltantes
2.	Codificación de variables categórica
3.	Transformación de variables
4.	Discretización
5.	Valores atípicos
6.	Magnitud de variables
7.	Variables de fecha y hora
8.	Creación de nuevos atributos

### 1. Imputación de datos faltantes

Imputación es el acto de sustituir los datos que faltan por estimaciones estadísticas de los mismos. El objetivo de 
de la imputación es producir un conjunto de datos completo que pueda utilizarse para entrenar modelos de 
aprendizaje automático.

Existen varias técnicas para la imputación de datos faltantes:

1.	Análisis de casos completos
2.	Imputación con la media / mediana / moda 
3.	Imputación con muestras aleatorias
4.	Sustitución por un valor arbitrario
5.	Imputación al final de la distribución
6.	Indicador de valores faltantes

![walking]({{ site.baseurl }}/assets/images/imputacion.png)


#### 1.1 Análisis de casos completos

En el análisis de casos completos utilizamos sólo las observaciones del conjunto de datos que contienen valores en 
todas las variables. En otras palabras, en el análisis de casos completos eliminamos aquellas observaciones que tengan 
valores faltantes en cualquiera de las variables.

Este procedimiento es adecuado cuando hay pocos datos faltantes. Si el set de datos contiene datos que faltan en 
múltiples variables, o algunas variables contienen una alta proporción de datos faltantes, con este procedimiento, podríamos 
eliminar fácilmente una gran parte de los de datos, y esto no es lo que queremos.

#### 1.2 Imputación con la media/mediana/moda

Podemos sustituir los valores faltantes por la media, la mediana o la moda de la variable. La imputación con la 
media/mediana/moda se muy utilizada tanto en organizaciones y como en competencias de datos.

Aunque en la práctica esta técnica se utiliza en casi todas las situaciones, el procedimiento es adecuado si los datos 
faltan al azar y en pequeñas proporciones. Sin embargo, si hay muchas observaciones faltantes, distorsionaremos la 
distribución de la variable, así como su relación con otras variables del conjunto de datos. La distorsión de la 
distribución de la variable puede afectar al rendimiento de los modelos lineales.

En el caso de las variables categóricas, la sustitución por la moda también se conoce como sustitución por la categoría 
más frecuente.

### 1.3 Imputación con muestras aleatorias

La imputación con muestras aleatorias se refiere a la selección aleatoria de valores de la variable para sustituir los 
datos que faltan. Esta técnica preserva la distribución de la variable y es adecuada cuando los datos faltan al azar. 
Sin embargo, hay que tener en cuenta la aleatoriedad del procedimiento, estableciendo adecuadamente una semilla. De lo 
contrario, el mismo valor faltante podría ser reemplazado por diferentes valores en diferentes ejecuciones de código, 
conduciendo a predicciones diferentes. Esto no es deseable cuando se utilizan nuestros modelos dentro de una organización.

#### 1.4 Sustitución por valor arbitrario

La sustitución por un valor arbitrario, como su nombre indica, se refiere a la sustitución de los datos que faltan por 
cualquier valor determinado arbitrariamente.

El reemplazo por un valor arbitrario es adecuado si los datos no faltan al azar, o si hay una gran proporción de valores 
faltantes. Si todos los valores son positivos, un reemplazo típico es utilizando -1. Alternativamente, el reemplazo por 999 o -999 
es una práctica común. Tenemos que prever que estos valores arbitrarios no sean una ocurrencia común en la variable. 
Normalmente elegimos valores mucho mas grandes o mucho mas chicos que los valores normales de la variable.

La sustitución por valores arbitrarios suele no ser adecuada para los modelos lineales, ya que lo más probable es que 
distorsione la distribución de las variables y, por tanto, no se cumplan los supuestos del modelo.

En el caso de las variables categóricas, este procedimiento equivale a sustituir las observaciones que faltan por la 
categoría "Faltante", un procedimiento ampliamente adoptado.

#### 1.5 Imputación al final de la distribución

La imputación al final de la distribución consiste en sustituir los valores faltantes por un valor situado en el 
extremo de la distribución de la variable. Esta técnica es similar a la imputación por un valor arbitrario. 
Sin embargo, al situar el valor al final de la distribución, no es necesario analizar la distribución de cada variable 
individualmente, ya que el algoritmo lo hace automáticamente por nosotros. Esta técnica de imputación suele funcionar 
bien con los algoritmos basados en árboles, pero puede afectar al rendimiento de los modelos lineales, ya que distorsiona 
la distribución de las variables.

#### 1.6 Indicador de dato faltante

El indicador de dato faltante es una variable binaria que indicar si el valor falta para una determinada observación. 
Esta variable toma el valor 1 si el valor de la observación falta, o 0 en caso contrario.
Hay que tener en cuenta que luego del indicador, aun hay que sustituir los valores que faltan en la variable original, 
lo que solemos hacer con la imputación con la media o la mediana. Al utilizar estas dos técnicas juntas, si el valor que 
falta tiene poder predictivo, será capturado por el indicador de dato faltante, y si no lo tiene, será enmascarado por 
la imputación con la media o la mediana. Estas dos técnicas combinadas suelen funcionar bien con los modelos lineales.

Hay que saber que al añadir indicadores de datos faltantes, podriamos incrementar bastante el número de variables y, 
como los atributos originales suelen tener valores ausentes en las mismas observaciones, muchos de estos indicadores 
terminan ser idénticos o estando altamente correlacionadas.

#### 1.7 Otras técnicas de imputación

Existen, además, técnicas de imputación multivariada, como MICE (Imputación multivariada de 
ecuaciones en cadena) que no cubriré en este artículo, pero que se tratan en el curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

### 2. Codificación categórica

La codificación de variables categóricas incluye técnicas utilizadas para transformar las cadenas 
de caracteres o etiquetas de las variables categóricas en números. Existen varios procedimientos:

1.	La codificación one-hot
2.	Codificación de recuento y frecuencia
3.	Codificación con la media de la variable objetivo
4.	Codificación ordinal
5.	Peso de la evidencia
6.	Codificación de etiquetas raras
 

![walking]({{ site.baseurl }}/assets/images/codificacion.png)


#### 2.1 Codificación one-hot

La codificación one-hot crea una variable binaria por cada una de las categorías de la variable. Estas 
variables binarias toman el valor 1 si la observación muestra una determinada categoría o 0 en caso contrario.

La codificación one-hot es adecuada para los modelos lineales. Sin embargo, incrementa el número de variables de forma 
drástica si las variables categóricas son muy cardinales o si hay muchas variables categóricas. Además, muchas de las 
variables derivadas podrían estar muy correlacionadas.

Podemos hacer codificación one-hot sólo de las categorías más frecuentes, y así controlar el incremento de las variables.

#### 2.2 Codificación de recuentos y frecuencias

En la codificación de recuento sustituimos las categorías por el recuento de las observaciones que muestran esa 
categoría en el conjunto de datos. Del mismo modo, podemos sustituir la categoría por la frecuencia -o el porcentaje- 
de las observaciones en el conjunto de datos. Por ejemplo, si 10 de nuestras 100 observaciones muestran el color azul, 
sustituiremos el valor azul por 10 si hacemos la codificación de recuento, o por 0,1 si lo sustituimos por la frecuencia.

Estas técnicas capturan la representación de cada categoría en un conjunto de datos y, son métodos muy populares en las 
competencias de ciencia de datos.

#### 2.3 Codificación con la media del target

En la codificación con la media del target, sustituimos cada categoría por el valor medio de la variable objetivo o target. Por 
ejemplo, tenemos la variable categórica "ciudad", y queremos predecir si el cliente comprará un televisor siempre que 
le enviemos una carta. Si el 30% de los habitantes de la ciudad "Londres" compran el televisor, sustituiríamos Londres 
por 0,3.

Esta técnica tiene 3 ventajas:

1.	No incrementa el número de variables.
2.	Capta cierta información sobre el target al codificar las categorías.
3.	Crea una relación monotónica entre la variable codificada y el target.

Las relaciones monotónica entre la variable y el target tienden a mejorar el rendimiento de los modelos lineales.

#### 2.4 Codificación ordinal

En la codificación ordinal sustituimos las categorías por dígitos, ya sea de forma arbitraria u ordenada. Si 
codificamos las categorías de forma arbitraria, asignamos un número entero por categoría de 1 a n, donde n es el 
número de categorías únicas. Si, por el contrario, asignamos los enteros de manera ordenada, obtenemos primero el valor
medio del target para cada categoría, luego ordenamos las categorías de 1 a n, asignando 1 a la categoría con el valor medio 
del target más alto, y n a la categoría con el valor medio más bajo.

La codificación ordinal ordenada tiene ventajas similares a la codificación con la media del target, es decir, no 
incrementa el número de variables y crea una relación monotónica entre la variable codificada y la variable objetivo.

#### 2.5 Peso de la evidencia

El peso de la evidencia (WOE del inglés) es una técnica utilizada para codificar variables categóricas para 
clasificación unicamente.

El WOE se determina de la siguiente manera:

Calculamos el porcentaje de casos positivos para cada categoría sobre el total de casos positivos. Por ejemplo, 20 casos 
positivos en la categoría A sobre el total de 100 casos positivos equivale al 20%. A continuación, calculamos el 
porcentaje de casos negativos en cada categoría respecto al total de casos negativos, por ejemplo, 5 casos negativos en 
la categoría A de un total de 50 casos negativos es igual al 10%. Luego, calculamos el WOE dividiendo los 
porcentajes de casos positivos y negativos, y tomando el logaritmo: para la categoría A en nuestro ejemplo 
el WOE sería igual a log(20/10).

WOE tiene la propiedad de que su valor será 0 si el fenómeno es aleatorio; será mayor que 0 si la probabilidad de que el 
target sea 0 es mayor, y será menor que 0 cuando la probabilidad de que el target sea 1 sea mayor.

La transformación WOE crea una clara representación visual de la variable, porque al mirar la variable codificada con el WOE, 
podemos ver, categoría por categoría, si favorece el resultado de 0, o de 1 del target. Además, WOE crea una relación 
monotónica entre la variable codificada y el target, y deja todas las variables dentro del mismo rango de valores.

#### 2.6 Codificación de etiquetas raras

Cuando nuestros datos tienen categorías raras o poco frecuentes, a menudo las encontramos sólo en el conjunto de 
entrenamiento o en el conjunto de prueba, pero no en ambos. Por lo tanto, es difícil trabajar con ellas. Si están en el 
conjunto de entrenamiento, pueden provocar un sobreajuste. Si están en el conjunto de pruebas, no sabremos realmente cómo 
codificarlas, ya que todos los parámetros de codificación se aprenden sólo para las categorías del conjunto de entrenamiento.

Por esto, las categorías que sólo están presentes en una pequeña proporción de las observaciones, tienden a agruparse en una 
categoría general como "Otros" o "Raro". Este procedimiento tiende a mejorar la generalización del modelo de aprendizaje 
automático, en particular para los métodos basados en árboles, y también la operacionalización de los modelos en producción.
 
![walking]({{ site.baseurl }}/assets/images/etiqueta_rara.png)


#### 2.7 Otros procedimientos de codificación

Existen otros métodos de codificación categórica, como la codificación binaria y el hash de variables, que no 
cubriré en este artículo, pero que se tratan en el curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

### 3. Transformación de variables

Algunos modelos de aprendizaje automático pueden beneficiarse de una distribución más homogénea de los valores de las 
variables. Si las variables no muestran una distribución normal, podemos aplicar una transformación matemática para "forzar" esta 
distribución.

Las transformaciones matemáticas más utilizadas son:

1.	Transformación logarítmica - log(x)
2.	Transformación recíproca - 1 / x
3.	Transformación de la raíz cuadrada - sqrt(x)
4.	Transformación exponencial - exp(x)
5.	Transformación Box-Cox
6.	Transformación Yeo-Johnson
 
![walking]({{ site.baseurl }}/assets/images/transformacion.png)


Box-Cox y Yeo-Johnson son variantes de las transformaciones exponenciales que abarcan varios exponentes y, 
por lo tanto, es más probable que consigan el resultado deseado.

Al aplicar las transformaciones matemáticas debemos tener en cuenta los valores de las variables. Por ejemplo, el 
logaritmo y la raíz cuadrada sólo admiten valores positivos, la transformación recíproca no está definida para 0 y la 
Box-Cox sólo está definida para valores positivos.

### 4. Discretización

La discretización se refiere a la organización de los valores de la variable en bins o intervalos. Hay múltiples formas de 
discretizar las variables:

1.	Discretización de igual distancia
2.	Discretización de igual frecuencia
3.	Discretización mediante árboles de decisión
4.	Discretización K-Means

![walking]({{ site.baseurl }}/assets/images/discretizacion.png)

#### 4.1 Discretización de igual distancia

En la discretización de igual distancia, los bins o límites de los intervalos se determinan de forma que cada intervalo 
tenga la misma distancia. Esto se consigue restando el valor mínimo del valor máximo de la variable, y dividiendo ese 
rango en la cantidad de intervalos deseados, digamos 10. A continuación, distribuimos las observaciones en esos intervalos. 
Ten en cuenta, que si la distribución es sesgada, esta transformación no mejora la dispersión de los valores.

#### 4.2 Discretización a igual frecuencia

En la discretización de igual frecuencia, los límites de los intervalos se determinan de forma que cada intervalo 
contenga el mismo número de observaciones. Esta es una mejor solución si queremos repartir los valores uniformemente en 
todos los intervalos. Se utilizan los percentiles para determinar los intervalos.

#### 4.3 Discretización con árboles de decisión

La discretización con árboles de decisión consiste en agrupar las observaciones segun las hojas finales de un árbol de 
decisión. Las distintas hojas contendrán un número diferente de observaciones, por lo que no se preserva la frecuencia 
como en la discretización de igual frecuencia. Además, cada nodo no es en sí mismo un intervalo, sino un valor de predicción.

La discretización con árboles de decisión puede mejorar el rendimiento del modelo al crear relaciones monótonas que ya 
capturan parte del poder predictivo de la variable. Esta técnica de ingeniería se utilizó en una competencia de datos en 
(KDD 2009).

#### 4.4 Discretización con K-medias

La discretización con K-medias consiste en agrupar las observaciones en clusters determinados por el algoritmo K-medias. La 
idea es agrupar las observaciones que son similares. El inconveniente es que necesitamos saber a priori cuántos clusters 
significativos tienen nuestras variables.

### 5. Valores atípicos

Los valores atípicos son valores inusualmente altos o inusualmente bajos con respecto al resto de las observaciones de la 
variable. Mantener o eliminar los valores atípicos depende de si son realmente predictivos o sólo un 
artefacto del procedimiento de medición que utilizamos para recoger los datos de la variable.

Por ejemplo, si queremos detectar fraude, o una enfermedad rara, estas ocurrencias pueden mostrar también valores inusuales 
en otras variables. Probablemente no queramos descartar esas observaciones.

Pero también existen casos, en los que el valor atípico es sólo un artefacto de medición, y podemos prescindir de él.
Si estamos seguros de que los valores atípicos no son útiles, podemos eliminarlos, censurarlos o discretizar la variable.

#### 5.1 Eliminación de valores atípicos

La eliminación de valores atípicos consiste en eliminar las observaciones atípicas del conjunto de datos. Los valores 
atípicos, por naturaleza, no son abundantes, por lo que este procedimiento no debería distorsionar el conjunto de datos 
de forma drástica. Pero si hay valores atípicos en muchas de las variables, podemos acabar eliminando una gran parte del 
conjunto de datos.

#### 5.2 Censura

La Winsorización o censura de valores atípicos, consiste en limitar los valores máximos y mínimos de una variable a un 
valor predefinido. Este valor predefinido puede ser arbitrario o puede derivarse de la distribución de la variable.

¿Cómo podemos obtener los valores máximos y mínimos? Si la variable está distribuida normalmente, podemos limitar los 
valores máximos y mínimos a la media más o menos 3 veces la desviación estándar. Si la variable es asimétrica, podemos 
utilizar la regla de proximidad del rango inter-cuartil o limitarla a los percentiles superior e inferior. En la 
Winsorización, utilizamos los percentiles para elegir los valores máximos y mínimos, normalmente el percentil 10 y el 90.

#### 5.3 Discretización

La discretización maneja los valores atípicos de forma automática, ya que los valores atípicos se colocan en los intervalos 
terminales junto con las demás observaciones de mayor o menor valor.

### 6. Escala de características

Muchos algoritmos de aprendizaje automático son sensibles a la magnitud de las variables, por lo que es una práctica común 
tener a todas las variables en una misma escala. Hay múltiples formas de escalar atributos, pero las más utilizadas son 
la normalización y la escala al mínimo y máximo valor.

#### 6.1 Normalización

La estandarización de variables implica restar la media de cada valor y dividir el resultado por la desviación estándar. 
La estandarización de variables hace que las variables tengan una media de valor 0 y una varianza unitaria, y es adecuada 
si las variables se distribuyen normalmente.

#### 6.2 Escala al mínimo y máximo valor

La escala al mínimo y máximo valor consiste en reescalar la variable a 0-1, lo que se consigue restando el mínimo de 
cada valor y dividiendo el resultado por el rango de valores. El rango de valores se calcula como el valor máximo menos 
el mínimo. La escala al mínimo y máximo valor ofrece una buena alternativa a la estandarización cuando las variables 
son sesgadas.

#### 6.3 Escala máxima absoluta

En este procedimiento dividimos cada valor por el valor máximo de la variable.

#### 6.4 Escalado robusto

El escalado robusto consiste en eliminar la mediana de cada valor y dividir el resultado por el rango inter-cuartil, que 
viene dado por la diferencia entre los percentiles 75 y 25. El procedimiento es similar a la escala al mínimo y 
máximo valor, pero ofrece una mejor dispersión de los valores para las variables muy sesgadas.

#### 6.5 Normalización con la media

En la normalización con la media, restamos de cada valor el valor medio y dividimos el resultado por el rango de valores, es decir, 
la diferencia entre el valor máximo y el mínimo.

#### 6.6 Escala a la unidad de longitud

El escalado a la longitud unitaria se refiere a la transformación de los datos de manera que el vector completo 
de la observacion tenga longitud uno. Al escalar a la longitud unitaria, dividimos cada valor de la variable por la distancia 
euclidiana o la norma de la observación (utilizando todas las variables).

Fijate que en este proceso de escalado, no escalamos variable por variable, sino observación por observación.

### 7. Ingeniería de fechas y horas

Podemos extraer un gran número de atributos de variables de fecha y hora. Algunos ejemplos son:

1.	año, semestre, trimestre, mes, semana
2.	día del mes, día de la semana, es fin de semana?
3.	hr, es la mañana, es la tarde, entre otros.

Con los valores anteriores podemos tambien determinar las vacaciones de verano y los días festivos, por ejemplo.
Además, podemos extraer información combinando variables de fecha y hora, por ejemplo, podemos obtener la 
edad a partir de la fecha de nacimiento y la fecha de la transacción, o el tiempo transcurrido entre 2 fechas, por nombrar algunas.

Un consejo: el paquete de Python Feature-engine extrae atributos automáticamente a partir de fechas y horas. 
Consulta el transformador [DatetimeFeatures](https://feature-engine.readthedocs.io/en/latest/user_guide/datetime/DatetimeFeatures.html).

### 8. Creación de atributos

Podemos crear nuevas variables, combinando las variables existentes con funciones matematicas como la suma, resta, media, 
desvio estandar y mas. También podemos realizar combinaciones polinómicas y otras transformaciones no lineales.

Para crear de variables muchas veces usamos el conocimiento de los datos para derivar nuevos 
atributos que sean comprensibles para los usuarios del modelo. En las competiciones 
de datos, cualquier enfoque de fuerza bruta para crear variables que no son necesariamente comprensibles puede darnos una 
ventaja.
 
La creación de variables se ve comúnmente en el Procesamiento del Lenguaje Natural, al crear bolsas de palabras o tablas de 
frecuencia a partir de texto.

### Para terminar

Si has llegado hasta aquí, gracias por leer. Espero que este articulo te haya sido util.

Si quieres saber más sobre la ingeniería de variables consulta nuestro curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning), o el paquete 
de código abierto [Feature-engine](https://feature-engine.readthedocs.io/en/latest/) (ojo, la documentación está en inglés).

Para saber más sobre nuestros otros cursos, visita [Entrena en Datos](https://entrenaendatos.teachable.com/).

### Referencias

•	[Ingeniería de variables para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning) - Curso

•	[Feature-engine](https://feature-engine.readthedocs.io/en/latest/): Paquete de Python para ingeniería de variables.
