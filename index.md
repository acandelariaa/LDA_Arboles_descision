# 🪐 Clasificación de habitabilidad basado en temperatura de equilibrio de exoplanetas
** Analisis de habitabilidad termica utilizando modelos de LDA y Arboles de decisión**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## Descripción

Este proyecto utiliza modelos de LDA y arboles de decisión para predecir la **temperatura de equilibrio** de exoplanetas y determinar cuáles podrían ser potencialmente habitables según criterios térmicos (250-300 K, compatible con agua líquida).

**Dataset**: NASA Exoplanet Archive (PSCompPars)
**Objetivo**: realizar una comparación de modelos LDA y Arboles de decisión para poder clasificar planetas

Recursos

| Dataset |[NASA_exoplanets (ya imputada)](dataset_final_exoplanetas.csv) |
|---|---|
| **Notebook** | [.ipynb](A2_2_LDA_Arboles.ipynb) |

---


## Metodología

- Definición de la variable de salida con dos o tres clases.

- Análisis del balance entre clases.

- División del dataset en entrenamiento y prueba, conservando la proporción de clases.

- Construcción del modelo LDA, seleccionando variables acordes a sus supuestos.

- Visualización e interpretación de las funciones discriminantes y la separación entre clases.

- Construcción del modelo de árbol de decisión con selección y justificación de parámetros.

- Aplicación de poda mediante el parámetro α para controlar la complejidad.

- Evaluación de ambos modelos con métricas de desempeño.

- Comparación e interpretación crítica de resultados cuantitativos y geométricos.

---

## Procedimiento


[Particion de datos](particion.md)

[LDA](LDA.md)

[Arboles](arboles.md)


## Referencias

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- Kopparapu et al. (2013). *Habitable Zones Around Main-Sequence Stars*
- Kasting et al. (1993). *Habitable Zones around Main Sequence Stars*

---



**By** 

*Juan Angel Candelaria Rodriguez*

Universidad de Monterrey | Inteligencia Artificial  

