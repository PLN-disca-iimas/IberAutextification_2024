## Información sobre los códigos

- En el archivo [Texto_a_graphs.ipynb](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/Texto_a_graphs.ipynb) se puede encontrar el codigo para .....

- En el archivo [Lista con métricas por texto.ipynb](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/Lista%20con%20m%C3%A9tricas%20por%20texto.ipynb) se encuentra el código que transforma texto a grafo, para utilizarse como objeto de [Networkx](https://networkx.org/) de tal forma que se obtienen distintas métricas como coeficiente de *clustering*, *dregree_centrality*, etc. Para finalmente tener como resultado un archivo .csv con todas esas características para un posterior análisis y/o entrenar algún modelo. NOTA: en este código se utilizan los archivos train/test previamente divididos.

- En el archivo [RN_task1_autextification2024](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/RN_task1_autextification2024.ipynb) se encuentra el código de la red neuronal simple, el cual procesa los embeddings de entrada por varias transformaciones (capas densas, activaciones y dropouts) y da predicciones de salida basadas en la capa final (en este caso sigmoid, ya que es una tarea binaria). NOTA: Es necesario crrerlo en el GPU del control ya que la RAM de las laptops o cumputadoras no es suficiente para correr la historia.

- En el archivo [XGB-LR-RF.ipynb](https://github.com/YaraHR/Modelos-de-procesamiento-de-lenguaje-natural-SS-/blob/main/XGB-LR-RF.ipynb) se encuentra el código con distintos modelos de clasificación (*XGBoost*, *Logistic Regression* y *Random Forest*) los cuales han sido entrenados con métricas de grafos (obtenidas con el [código](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/Lista%20con%20m%C3%A9tricas%20por%20texto.ipynb)) y estilométricas ([código](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/stylometry_train_test_S1.ipynb)), para determinar si un texto es generado por máquina o por humano. Los resultados son:

|                Modelo               | F1-Score |
|-------------------------------------|----------|
|    Random Forest Classifier (RFC)   | **0.719** |
|        LogisticRegression (LR)      | 0.712 |
| Extreme Gradient Boosting (XGBOOST) | 0.638 |
  


