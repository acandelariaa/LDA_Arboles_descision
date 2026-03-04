### Definición del problema y partición de los datos

En este trabajo se comparan dos enfoques de clasificación supervisada: Linear Discriminant Analysis (LDA) y árboles de decisión, los cuales parten de supuestos y estructuras diferentes. Mientras LDA construye fronteras lineales bajo supuestos probabilísticos sobre la distribución de las clases, los árboles de decisión generan particiones jerárquicas sin asumir linealidad. El objetivo es analizar el comportamiento geométrico y el desempeño predictivo de ambos modelos sobre el mismo conjunto de datos, para determinar cuál resulta más adecuado para el problema planteado.

En este caso de utilizara una base de datos relacionada a los exoplanetas, con el fin de clasificar si estos son habitables o no basado en variables dentro del dataset.

Tomaremos como referencia la temperatura de equilibrio `pl_eqt` de estos planetas, si se asemejan a la temperatura de equilibrio de la tierra **255 K**, o se encuentran en el rango de **250 - 300 K**, podemos considerarlos habitables.



> Python Code



```python
#Cargar datos
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/Inteligencia_Artificial_1/dataset_final_exoplanetas.csv")
```


> Output



| pl_name                       | pl_eqt | pl_insol   | st_teff | st_rad   |
|--------------------------------|--------|------------|---------|----------|
| 11 Com b                       | 1700.0 | 792.57818  | 4060.0  | 1.313160 |
| 11 UMi b                       | 1450.0 | 372.74340  | 3100.0  | 0.840000 |
| 14 And b                       | 1600.0 | 513.74032  | 3530.0  | 0.366450 |
| 14 Her b                       | 2369.0 | 427.54252  | 2943.0  | 0.289776 |
| 16 Cyg B b                     | 2677.0 | 5262.59620 | 7244.6  | 1.976258 |
| 17 Sco b                       | 2506.0 | 335.96392  | 3021.0  | 0.600455 |
| 18 Del b                       | 1250.0 | 169.58952  | 2825.0  | 0.218870 |
| 1RXS J160929.1-210524 b         | 1616.0 | 513.74032  | 3592.0  | 0.672832 |
| 24 Boo b                       | 1600.0 | 468.74252  | 3352.0  | 0.437589 |
| 24 Sex b                       | 700.0  | 151.06598  | 7295.0  | 1.487260 |



> Nota: la base de datos utilizada, ya se ha utilizado en tareas y proyectos anteriores, con la finalidad de no tener que buscar multiples bases de datos para cada tarea, se va a trabajar con esta base de datos, la cual ya esta limpia e imputada.


### Crear variable binaria de habitabilidad

Como en este dataset tenemos mas que nada numeros, como antes comentamos, tomaremos en cuenta la temperatura de equilibrio del planeta para clasificarlo como habitable o no, de modo que crearemos una columna Habitability la cual tendra, `1` si cae en el rango de [250 - 300] Kelvin y si no cae dentro del rango, sera `0`, es decir, no habitable 

> Python Code

```python
# Crear variable binaria usando operaciones vectorizadas
df["Habitability"] = ((df["pl_eqt"] >= 250) & (df["pl_eqt"] <= 300)).astype(int)

# Verificar distribución
print(df["Habitability"].value_counts())
```

> Output


```text
Habitability
0    4488
1      78
Name: count, dtype: int64
```


### Balance / Desbalance de clases

Con lo anterior podemos ver que claramente hay un desbalance de clases, por muchosimo mas, ya que al parecer tenemos 4488 planetas NO habitables y 78 planetas habitables, de modo que tendremos una proporcion muy pequeña de planetas habitables

### Calcular proporción



> Python Code


```python
# Proporciones
print(df["Habitability"].value_counts(normalize=True) * 100)
```

> Output



```text
Habitability
0    98.291721
1     1.708279
Name: proportion, dtype: float64
```



>IMPORTANTE: Bastante pequeña la proporción de planetas habitables, esto podria representar un problema ya que el modelo podria siempre predecir planetas que NO son habitables y confundir los pocos casos que si lo sean, de modo que podria ser necesario revisar precisión, F1 , matriz de confusión, entre otros.


### Dividir datos


En este caso, como tenemos un desbalance en las clases, vamos a utilizar el comando stratify, de esa forma aseguramos que los datos de train y test tengan la misma proporción en medida de lo posible.



> Python Code

```python
from sklearn.model_selection import train_test_split

# Variables predictoras (X)
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Habitability"])

# Variable objetivo (y)
y = df["Habitability"]

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,   # mantiene proporción de clases
    random_state=42
)

# Verificar balance en cada subconjunto
print("Train:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest:")
print(y_test.value_counts(normalize=True) * 100)

```

> Output

```text
Train:
Habitability
0    98.279099
1     1.720901
Name: proportion, dtype: float64

Test:
Habitability
0    98.321168
1     1.678832
Name: proportion, dtype: float64

```

Ya que tenemos nuestros datos preparados, crearemos un modelo de Linear Discriminant Analysis y veremos su comportamiento

----

[Siguiente pagina >>>](LDA.md)


