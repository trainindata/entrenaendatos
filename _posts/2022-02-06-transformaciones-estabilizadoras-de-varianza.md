---
layout: post
title:  "Transformaciones estabilizadoras de varianza"
author: sole
categories: [feature engineering, machine learning ]
image: assets/images/Introimage.png
---

Probablemente hayas oído que un paso común antes de entrenar modelos de aprendizaje automático es transformar las variables sesgadas para cambiar su distribución a algo más cercano a la distribución normal. Pero, ¿por qué hacemos esto? ¿Qué variables debemos transformar? ¿Qué transformaciones debemos utilizar? Y, ¿necesitamos transformar las variables para entrenar cualquier algoritmo de aprendizaje automático?
Estas son las preguntas que abordaremos en este artículo. Empecemos.

## Serie: Ingeniería de variables para aprendizaje automático

Este artículo forma parte de una serie de artículos sobre ingeniería de variables para datos tabulares. Estás leyendo 
el artículo nº 4. Puedes encontrar otros artículos en los siguientes enlaces:

1.	[Ingeniería de variables para machine learning](https://www.entrenaendatos.com/ingenier%C3%ADa-de-variables-para-machine-learning/)
2.	Imputación de datos faltantes
3.	Codificación de variables categóricas
4.	**Transformaciones estabilizadoras de varianza**
5.	Magnitud y escalamiento de las variables
6.	Discretización
7.	Creación de nuevos atributos
8.	[Implementación en Python de técnicas ingeniería de variables](https://www.entrenaendatos.com/implementación-en-Python-técnicas-ingeniería-de-variables/)

Si quieres saber más sobre estas y otras técnicas de ingeniería de variables, consulta nuestro curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

Empecemos.

### ¿Por qué transformamos las variables?

Muchos de los métodos estadísticos utilizados en el análisis de datos hacen suposiciones sobre los datos. Por ejemplo, 
para sacar conclusiones de un modelo de regresión lineal, muchos supuestos deben ser ciertos. Algunos de los supuestos 
son los siguientes:

•	Los valores de la variable dependiente (es decir, el objetivo) son independientes.
•	Existe una relación lineal entre el objetivo y las variables independientes.
•	Los residuos, es decir, la diferencia entre las predicciones y los valores reales del objetivo, se distribuyen normalmente y se centran en cero.

Muchas personas, entre las que me incluyo, confunden este último supuesto con la idea de que todas las variables 
independientes tienen que estar distribuidas normalmente. Pero no es así. Lo que tiene que estar normalmente distribuido 
y centrado en cero son los residuos, lo que significa que cualquier diferencia entre las predicciones y el objetivo es 
simplemente aleatoria, o en otras palabras, no puede ser explicada.

**¿Qué ocurre si no se cumplen los supuestos?**

Cuando los supuestos no se cumplen, las conclusiones probabilísticas extraídas del análisis de datos pueden no ser 
fiables. Afortunadamente, podemos corregir el fallo en los supuestos transformando las variables antes del análisis. 
Esto mejoraría el rendimiento y la fiabilidad de los modelos.

El ejemplo más obvio es transformar la propia variable objetivo cuando su distribución está sesgada. Sin embargo, la 
transformación de las variables independientes, muy a menudo ayuda a cumplir los supuestos del modelo cuando los datos 
originales no lo hacen. Las transformaciones que aplicamos a las variables independientes y al objetivo se denominan 
transformaciones estabilizadoras de las variables, y explicaré lo que significa en las próximas secciones.

Quizás ya hayas adivinado que la transformación de variables suele aplicarse cuando pretendemos analizar los datos 
mediante pruebas estadísticas lineales como ANOVA y cuando entrenamos modelos de regresión lineal generalizada. En 
otras palabras, no es necesario transformar las variables cuando se entrenan modelos no lineales como los algoritmos 
basados en árboles de decisión, vecinos más cercanos o redes neuronales.

### ¿Qué son las transformaciones estabilizadoras de varianza?

La transformación de variables consiste en sustituir los valores originales de las variables por una función de esa 
variable. La transformación de variables con funciones matemáticas ayuda a reducir la asimetría de las variables, 
mejorando así la dispersión de valores, y a veces desenmascara las relaciones lineales y aditivas entre los predictores 
y el objetivo.

Las transformaciones matemáticas más utilizadas son las de logaritmo, recíproca, potencia y raíz cuadrada, así como las 
de Box-Cox y Yeo-Johnson. Estas transformaciones se denominan comúnmente "transformaciones estabilizadoras de la varianza". 
Las transformaciones estabilizadoras de la varianza pretenden llevar la distribución de la variable a una forma más 
simétrica o, en otras palabras, gaussiana.
 
La distribución gaussiana:

![walking]({{ site.baseurl }}/assets/images/Gaussian.png)


Muchas transformaciones estabilizadoras de la varianza se discutieron y analizaron en el contexto de las distribuciones 
de Poisson, donde la varianza de la variable es igual a la media. Por lo tanto, cuanto mayor sea la media, mayor será la 
varianza. La transformación de las variables tiene como objetivo obtener valores, tales que su varianza sea independiente 
de su media. Por ello se refieren como “transformaciones estabilizadoras de la varianza”.
 
Distribución de Poisson (https://commons.wikimedia.org/wiki/File:Poisson_pmf.svg):

![walking]({{ site.baseurl }}/assets/images/Poisson_pmf.svg)

En los siguientes párrafos, discutiremos las siguientes transformaciones estabilizadoras de la varianza:

* Logaritmo
* Recíproca
* Raíz cuadrada
* Arcoseno
* Potencia
* Box-Cox
* Yeo-Johnson

Preparate.

### Logaritmo

La función logaritmo es una transformación útil para datos positivos con una distribución sesgada a la derecha (las 
observaciones se acumulan en los valores más bajos de la variable).

Variables candidatas para la transformación logarítmica son las variables continuas, como el salario, que tienden a 
mostrar una fuerte acumulación de observaciones hacia valores más pequeños. En otras palabras, la mayoría de la gente 
gana poco, sólo unos pocos ganan mucho. Ya está, lo hemos dicho.

En particular, si tomamos la variable Median Income del conjunto de datos de viviendas de California de scikit-learn, 
vemos que es continua y sesgada a la derecha:
 
![walking]({{ site.baseurl }}/assets/images/MedianIncRaw.png)


Sin embargo, tras la transformación con el logaritmo, observamos valores más repartidos y uniformes:

![walking]({{ site.baseurl }}/assets/images/MedIncLog.png)

 
### Transformación recíproca

La función recíproca, definida como 1/x, es una transformación con un efecto drástico en la distribución de la variable.

La transformación recíproca suele ser útil cuando tenemos cocientes, es decir, valores resultantes de la división de dos 
variables. Los ejemplos clásicos son variables como la densidad de población, es decir, las *personas por superficie*, o 
la ocupación de las viviendas, es decir, el *número de ocupantes por vivienda*.

Cuando "invertimos" variables como éstas, pasamos de una representación de *personas por área* a *área por persona*, o de 
*ocupantes por casa* a *casas por ocupante*. Las variables transformadas siguen teniendo sentido (para los humanos) y suelen 
mostrar una mejor distribución de los valores.

Si no me crees, echa un vistazo al histograma de la ocupación de las casas del conjunto de datos de viviendas de 
California de scikit-learn, una variable muy sesgada:

![walking]({{ site.baseurl }}/assets/images/HouseOccupancyRaw.png)

Y mira la distribución de la misma variable después de la transformación recíproca:
 
![walking]({{ site.baseurl }}/assets/images/HouseOccupancyReciprocal.png)


Puedes ver cómo la transformación recíproca mejoró drásticamente la dispersión de los valores e incluso transformó una 
variable discreta en una continua.

Ojo, la transformación recíproca no está definida para el valor 0. Así que si nuestras variables contienen ceros... 
bueno, deberíamos intentar otra cosa.

### Transformación con la raíz cuadrada

Hemos mencionado anteriormente que las transformaciones estabilizadoras de la varianza se discuten en el contexto de las 
distribuciones de Poisson. La transformación de raíz cuadrada, √x, así como sus variaciones, la transformación de 
Ascombe, √(x+3/8), y la transformación de Freeman-Tukey, √x + √(x+1) son transformaciones estabilizadoras de la varianza 
que transforman variables con una distribución de Poisson (recuentos, en otras palabras) en variables con una distribución 
gaussiana aproximadamente estándar. La transformación de raíz cuadrada es una forma de transformación de potencia en la 
que el exponente es 1/2 y sólo está definida para valores positivos. En los próximos párrafos hablaremos de las 
transformaciones de potencia generales.

Existen abundantes variables con distribuciones de Poisson, como el número de tarjetas de crédito o cuentas bancarias por 
persona, el número de hijos por familia o el número de mascotas. Esas variables son naturalmente de conteo. Podemos tener 
1 o 2 hijos, pero desde luego no 1 y medio. Y lo mismo ocurre con los otros ejemplos. Así que, en estos casos, una 
transformación de raíz cuadrada podría ser adecuada para estabilizar la varianza.

Ahora bien, estas variables no son continuas, por lo que no veremos los cambios obvios que observamos con las variables 
continuas después de la transformación, pero veremos en los próximos gráficos cómo la transformación de una variable con 
una distribución de Poisson devuelve observaciones más uniformemente distribuidas a lo largo de la diagonal en los 
gráficos Q-Q.

Ejemplo de distribución de Poisson antes de la transformación:
 
![walking]({{ site.baseurl }}/assets/images/Poisonraw.png)

Distribución teórica de Poisson - izquierda: histograma, derecha: Gráfico Q-Q.

La misma distribución después de la transformación de la raíz cuadrada:

![walking]({{ site.baseurl }}/assets/images/PoisonSquareRoot.png)
 
Distribución de Poisson tras la transformación de la raíz cuadrada - izquierda: histograma, derecha: Gráfico Q-Q.

Obsérvese cómo las observaciones se distribuyen más uniformemente a lo largo de la línea roja en la imagen precedente.

### Transformación Arcoseno

Antes de entrar en las transformaciones de potencia generalizadas, echemos un vistazo a la transformación arcoseno. La 
transformación arcoseno, también llamada transformación de raíz cuadrada arcoseno, o transformación angular, tiene la 
forma de arcoseno(sqrt(x)) donde x es un número real entre 0 y 1.

La transformación de raíz cuadrada arcoseno ayuda a tratar las probabilidades, los porcentajes y las proporciones. Su 
objetivo es estabilizar la varianza de la variable y devolver valores más uniformes (de aspecto gaussiano).

Como puedes imaginar, hay muchos ejemplos de variables que podrían ser candidatas para la transformación arcoseno, como 
las del conjunto de datos de cáncer de mama de scikit-learn.

Puedes ver cómo un montón de estas variables muestran distribuciones sesgadas en su estado original:

![walking]({{ site.baseurl }}/assets/images/breast_cancer_raw.png)
 
Y después de la transformación arcoseno, los valores se distribuyen más uniformemente:

![walking]({{ site.baseurl }}/assets/images/breast_cancer_arcsin.png)

 Probablemente esta era una transformación que estaba fuera de tu radar, y a decir verdad, rara vez se utiliza. Pero aquí 
 está, una función que ha sido ampliamente estudiada en el pasado.
 
### Transformaciones de potencia

Las funciones de potencia son formulaciones matemáticas como ésta X = X^lambda donde lambda puede tomar cualquier valor. 
Las transformaciones de raíz cuadrada y cúbica son casos especiales de transformaciones de potencia donde lambda es 1/2 o 
1/3, respectivamente. La transformación recíproca es también una transformación de potencia donde lambda es -1. Así que, 
¡todo el tiempo hemos estado hablando de transformaciones de potencia!

El desafío de elegir una transformación de potencia reside en encontrar un valor adecuado para el parámetro lambda que 
devuelva variables cuyos valores estén más uniformemente distribuidos.

En los párrafos anteriores hemos hablado de los casos especiales de la raíz cuadrada y la transformación recíproca porque 
son adecuados para variables específicas. En realidad, no probamos manualmente los exponentes para ver cuál funciona mejor, 
porque la transformación Box-Cox, de la que hablaremos en el siguiente párrafo, encuentra automáticamente lambda por 
nosotros. Pero como orientación general, si los datos están sesgados a la derecha, es decir, las observaciones se acumulan 
hacia los valores más bajos, utilizamos lambda <1 y si los datos están sesgados a la izquierda, es decir, hay más 
observaciones alrededor de los valores más altos, entonces utilizamos lambda > 1. Fin de la historia.

### Transformación Box-Cox

La transformación Box-Cox es una generalización de la familia de transformaciones de potencia, y se define por:
 
![walking]({{ site.baseurl }}/assets/images/boxcoxformula.png)

donde X es la variable y λ es el parámetro de transformación.

La transformación de Box-Cox incluye los casos especiales de transformaciones que discutimos anteriormente, incluyendo 
sin transformación (λ = 1), el logaritmo (λ = 0), el recíproco (λ = - 1), la raíz cuadrada (cuando λ = 0,5 aplica una 
versión escalada y desplazada de la función de raíz cuadrada) y la raíz cúbica.

En la transformación Box-Cox, se evalúan varios valores de λ utilizando la máxima probabilidad, y se selecciona la λ que 
devuelve la mejor transformación. Así que, como puede adivinar, la transformación de Box-Cox tiende a ser la opción 
preferida por los profesionales del aprendizaje automático, porque no necesitamos pensar (aunque pensar es realmente una 
buena práctica) qué transformación debemos aplicar a qué variable, o probar diferentes transformaciones manualmente.

La única advertencia de la transformación Box-Cox es que fue diseñada sólo para variables positivas. Por lo tanto, si sus 
variables contienen valores negativos, puede desplazar la distribución añadiendo una constante o utilizar la transformación 
de Yeo-Johnson.

### Transformación Yeo-Johnson

La transformación Yeo-Johnson es una extensión de la transformación Box-Cox que ya no está limitada a los valores 
positivos. En otras palabras, la transformación Yeo-Johnson puede utilizarse en variables con valores cero y negativos, 
así como con valores positivos.

La transformación de Yeo-Johnson se define como sigue:
 
![walking]({{ site.baseurl }}/assets/images/yeoj.png)

En resumen, si la variable X es estrictamente positiva, entonces, la transformación de Yeo-Johnson es la misma que la 
transformación de potencia de Box-Cox de X + 1. Si X es estrictamente negativa, entonces la transformación de Yeo-Johnson 
es la transformación de Box-Cox de (-X + 1) pero con potencia 2 - λ. Si la variable tiene valores positivos y negativos, 
entonces la transformación es una mezcla de estas 2 funciones, por lo que se utilizan potencias diferentes para los 
valores positivos y negativos de la variable. En mi opinión, es un poco lioso, pero mientras funcione...

### ¿Por qué, qué, cómo y cuándo transformar las variables?

Empezamos el articulo con las siguientes preguntas:

* ¿por qué transformamos las variables?
* ¿Qué variables debemos transformar?
* ¿Qué transformaciones debemos utilizar?
* Y, ¿necesitamos transformar las variables para entrenar cualquier algoritmo de aprendizaje automático?

A esta altura, creo que responder estas preguntas.

**¿Por qué transformamos las variables?**

Para que los datos cumplan los supuestos de determinados modelos estadísticos, normalmente modelos lineales, y poder así 
extraer conclusiones precisas o fiables del análisis de los datos.

**¿Qué variables debemos transformar?**

En general, las que muestran distribuciones sesgadas.

**¿Qué transformaciones debemos utilizar?**

Hay un montón de transformaciones que podemos utilizar. Si estamos interesados en entender nuestras variables transformadas, 
podríamos preferir hacer un análisis de los datos y seleccionar qué transformación aplicar a cada variable basándonos en 
lo que hemos discutido a lo largo de la entrada del blog. Por ejemplo, aplicaríamos la raíz cuadrada a los recuentos, el 
arcoseno a las fracciones y el recíproco a los cocientes. Aplicaríamos el logaritmo a las variables con observaciones que 
se acumulan en valores inferiores, y para todo lo demás, otras transformaciones de potencia.

En la práctica, para agilizar las cosas, nos limitamos a utilizar Box-Cox o Yeo-Johnson, que consideran todas las 
transformaciones anteriores, y eligen la transformación automáticamente. Pero cuidado, ¡la automatización no siempre 
resuelve el problema!

A veces, aplicar las transformaciones a ciegas crea un problema. Así que siempre es una buena práctica trazar las variables 
después de la transformación, y estar seguros de que hemos obtenido el resultado esperado. Lo sé, parezco la abuela.

Observa, por ejemplo, la siguiente figura extraída de la documentación de [scikit-learn](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html):

![walking]({{ site.baseurl }}/assets/images/sklearn.png)

En el gráfico, vemos que aplicar transformaciones a ciegas a variables que ya se distribuyen normalmente (lila en la figura) 
no cambia realmente la distribución, por lo que quizá queramos ahorrarnos ese paso innecesario en nuestro código o rutinas 
de aprendizaje automático.

Por otro lado, las transformaciones Box-Cox y Yeo-Johnson pueden no devolver distribuciones con forma gaussiana después 
de la transformación, como en los ejemplos extremos de las distribuciones bimodal (verde) y uniforme (negro).

¿A dónde quiero llegar con esto? Por mucho que sea tentador transformar automáticamente las variables y entrenar los 
modelos, siempre conviene analizar las transformaciones y entender qué datos tenemos en nuestros conjuntos de datos, y qué estamos pasandole a nuestros modelos, es decir, las variables después de la transformación.

**Por último, ¿necesitamos transformar las variables para entrenar cualquier algoritmo de aprendizaje automático?**

No. Estas transformaciones fueron estudiadas y diseñadas para su uso con modelos lineales. Por lo tanto, si quiere entrenar modelos no lineales como los algoritmos basados en árboles de decisión o vecinos más cercanos, es mejor que se salte este paso.

### Implementación en Python de las transformaciones estabilizadoras de la varianza

He hablado mucho, o mejor dicho, he escrito mucho sobre las transformaciones estabilizadoras de la varianza, pero no te 
he mostrado realmente cómo implementar estas transformaciones en Python, ¿verdad? 

Aplicar estas transformaciones con Python es sencillo. Podemos hacerlo con Numpy de la siguiente manera:

```
importar numpy como np
data["variable_log"] = np.log(data["variable_original"])
```

Para la transformación recíproca, usaríamos `np.reciprocal()`, para la raíz cuadrada `np.sqrt()`, y para la potencia 
`np.exp(data["variable_original"], lambda)`, donde lambda es el exponente deseado de la transformación.

Para BoxCox y Yeo-Johnson, utilizaríamos scipy.stats:

```
import scipy.stats as statsX_
tf["new_var"], param = stats.boxcox(X["original_var"])
X_tf["new_var"], param = stats.yeojohnson(X["original_var"])
```

Donde param es el lambda adecuado encontrado por la transformación.

Sin embargo, con Numpy y scipy.stats tenemos que modificar una variable cada vez. En realidad podemos modificar muchas 
variables simultáneamente utilizando scikit-learn, o la biblioteca Feature-engine.

Así, por ejemplo, si queremos aplicar la transformación Box-Cox con Feature-engine, haríamos lo siguiente:

```
from feature_engine.transformation import BoxCox
Transformerboxcox = BoxCoxTransformer()
boxcox.fit(X_train)
train_transformed = boxcox.transform(X_train)
test_tranformed = boxcox.transform(X_test)
```

Con Scikit-learn lo haríamos:

```
from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer(method="box-cox", standardize=False)
boxcox.fit(X_train)
train_transformed = boxcox.transform(X_train)
test_tranformed = boxcox.transform(X_test)
```

Hay diferencias entre las implementaciones de scikit-learn y Feature-eng
ía de variables](https://www.entrenaendatos.com/implementación-en-Python-técnicas-ingeniería-de-variables/)

También destaco las diferencias entre Numpy, scipy.stats, Scikit-learn y Feature engine en mi 
[curso online](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning).

### Para terminar...

Si quieres saber más sobre la ingeniería de variables consulta nuestro curso [Ingeniería de variables 
para machine learning](https://entrenaendatos.teachable.com/p/ingenieria-de-variables-para-machine-learning), o el paquete 
de código abierto [Feature-engine](https://feature-engine.readthedocs.io/en/latest/) (ojo, la documentación está en inglés).

Para saber más sobre nuestros otros cursos, visita [Entrena en Datos](https://entrenaendatos.teachable.com/).

Para el código que generó los gráficos mostrados en este artículo, y más detalles sobre la implementación en Python de 
las diversas transformaciones, visita [este repositorio](https://github.com/solegalli/Python-Feature-Engineering-Cookbook-Second-Edition/tree/main/ch03-variable-transformation).

Tanto scikit-learn como Feature-engine contienen una extensa documentación con implementaciones de código de las 
transformaciones estabilizadoras de la varianza. Echa un vistazo a 
[esta página de Feature-engine](https://feature-engine.readthedocs.io/en/latest/user_guide/transformation/index.html) o 
a [este enlace de scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution), 
y encontrarás todo lo que necesitas para transformar tus variables (en inglés, lo único).

Esto es todo lo que tengo que decir sobre las transformaciones estabilizadoras de la varianza. Espero que hayas encontrado 
el artículo útil, y si has llegado hasta aquí, gracias por leer.

Gracias por leer.
