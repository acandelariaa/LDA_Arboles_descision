### Arboles de decisión

En este apartado exploraremos la viabilidad de un modelo de arboles de decisión para la clasificación de habitabilidad de exoplanetas.



Bien, ya vimos que al parecer con LDA, el modelo tiene muchos problemas para predecir la habitabilidad de los planetas, esto muy relacionado al fuerte desbalance de las clases.

Debido a esto, podemos probar un tipo diferente de modelo, en este caso utilizaremos arboles de desición, para ver si empeora o mejora.

### Crear modelo

>Python Code


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Crear modelo base
tree = DecisionTreeClassifier(
    random_state=42
)

# Entrenar
tree.fit(X_train, y_train)

# Predicciones
y_pred_tree = tree.predict(X_test)
```

> Nota: se definió: una semilla para el random state

### Ver matriz de confusión y reporte de clasificación

>Python Code


```python
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_tree))
```

>Output


```text
Accuracy: 0.9992700729927008

Matriz de Confusión:
[[1346    1]
 [   0   23]]

Reporte de Clasificación:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1347
           1       0.96      1.00      0.98        23

    accuracy                           1.00      1370
   macro avg       0.98      1.00      0.99      1370
weighted avg       1.00      1.00      1.00      1370

```


> Nota: aqui podemos ver claramente que el modelo esta sobre ajustado ya que tenemos 100% de recall, F1 score y precisión.

Posiblemente esto haga que nuestros datos de test tengan resultados muy malos.

Una opción podria ser la poda de arboles, donde probemos distintos valores de alpha y ver cual se adecua mas a nuestro contexto. Basicamente probaremos valores de alpha hasta encontrar el mas optimo.\
