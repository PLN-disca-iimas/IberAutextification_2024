## Información sobre los códigos

- En el archivo [Texto_a_graphs.ipynb](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/Texto_a_graphs.ipynb) se puede encontrar el codigo para .....

- En el archivo [Lista con métricas por texto.ipynb](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/Lista%20con%20m%C3%A9tricas%20por%20texto.ipynb) se encuentra el código que transforma texto a grafo, para utilizarse como objeto de [Networkx](https://networkx.org/) de tal forma que se obtienen distintas métricas como coeficiente de *clustering*, *dregree_centrality*, etc. Para finalmente tener como resultado un archivo .csv con todas esas características para un posterior análisis y/o entrenar algún modelo.

- En el archivo [RN_task1_autextification2024](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/RN_task1_autextification2024.ipynb) se encuentra el código de la red neuronal simple, el cual procesa los embeddings de entrada por varias transformaciones (capas densas, activaciones y dropouts) y da predicciones de salida basadas en la capa final (en este caso sigmoid, ya que es una tarea binaria). NOTA: Es necesario crrerlo en el GPU del control ya que la RAM de las laptops o cumputadoras no es suficiente para correr la historia.
  

NOTA: el archivo [Lista con métricas por texto (adicionales).ipynb](https://github.com/PLN-disca-iimas/IberAutextification_2024/blob/main/Subtask_1/Otros%20Experimentos/Lista%20con%20m%C3%A9tricas%20por%20texto%20(adicionales).ipynb) hace lo mismo que el archivo anterior (Lista con métricas por texto.ipynb), solamente cambian las métricas por *number_edges*, *number_nodes*, etc.
