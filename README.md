# **IberAutextification 2024** 👩🏻 :robot: 
# Introducción 🔍
La nueva generación de los grandes modelos de lenguaje (LLMs) es capaz de generar textos fluidos, coherentes y plausibles en muchos idiomas, volviéndose difícil de diferenciar del texto escrito por humanos. Esto ha generado la preocupación de que los LLMs puedan ser utilizados con fines maliciosos, como la generación de noticias falsas, la difusión de opiniones polarizadas o el aumento de la credibilidad de campañas de phishing, entre otros. Estos ataques pueden realizarse en diferentes idiomas, dominios, modelos o estrategias de generación, lo que dificulta su moderación.

[IberAuTexTification](https://sites.google.com/view/iberautextification) surge de la necesidad de desarrollar estrategias de moderación efectivas y generalizables para enfrentar estos modelos, cada vez más sofisticados. Es la segunda versión de la tarea compartida AuTextification en IberLEF 2023, ampliada a tres dimensiones: más modelos, más dominios y más idiomas de la Península Ibérica.

En IberAuTexTification 2024 se introducen dos sub-tareas. La primera consiste en una clasificación binaria con dos clases: identificar si un texto fue escrito por un humano o generado por un LLM. La segunda sub-tarea es una clasificación multiclase en la que se busca identificar qué modelo generó el texto producido por un LLM. 

# Metodología 💡 
## Corpus 📄📄
Para evitar que nuestros modelos presenten overfitting, realizamos una mezcla aleatoria del corpus de cada subtarea antes de dividirlo en conjuntos de entrenamiento y prueba. La división se realiza de manera estratificada, de modo que el 70% de los datos se asigna al conjunto de entrenamiento y el 30% restante al conjunto de validación, asegurando particiones bien equilibradas en ambos conjuntos. 

El corpus se encuentra [aqui](https://drive.google.com/drive/folders/1VdTmKAzrfFrL-MKEmsvEXjYKugrm5Rw7?usp=share_link).
Ahí se puede almacenar cualquier cambio o procesamiento que se le haga al dataset. 

## Modelos 🧩🧩
El enfoque que nosotros adoptamos consiste en una arquitectura que incorpora Redes Neuronales de Grafos (GNN), Modelos Multilingües de Gran Escala (LLM) y características estilométricas. El diagrama general de la arquitectura de los modelos presentados en las subtareas compartidas se muestra a continuación

![Descripción de la imagen](https://drive.google.com/uc?export=view&id=1Zzm_o999lkIjJ1NZNQ_8NeghzvQORxaI)

Este repositorio presenta la arquitectura que emplea los LLMs junto con las características estilométricas, una de las tres arquitecturas propuestas en las subtareas. En este modelo, primero se realiza un fine-tuning a tres modelos de gran escala (BERT-Base-Multilingual, Multilingual-E5-Large, XLM-RoBERTa-Base). Luego, una vez que los LLMs han sido ajustados, se extraen los vectores embeddings de la última capa de cada modelo. Esto se hace con el objetivo de capturar toda la información contextual contenida en estos vectores y concatenarla con características estilométricas extraídas directamente del corpus original. Finalmente, el vector resultante alimenta un modelo de machine learning tradicional: Stochastic Gradient Descent (SGD) para la primer subtarea y Support Vector Classification (SVC) para la segunda. 

El artículo se puede ver a detalle en el siguiente [enlace](https://ceur-ws.org/Vol-3756/IberAuTexTification2024_paper7.pdf)

# Resultados 📊
Durante la etapa de desarrollo, se realizaron [experimentos](https://docs.google.com/spreadsheets/d/1uVSCHPzADm_dnnxsLgKpeQoZJ753lyxW9lIBk-bKAWM/edit?usp=sharing) con diferentes arquitecturas, comenzando inicialmente con modelos simples que consistían en utilizar únicamente modelos de machine learning tradicional. Al percatarse de que agregar características estilométricas y, por separado, que los vectores embeddings de los LLMs finetuneados mejoraban el rendimiento de estos modelos, dejamos de lado las arquitecturas simples y comenzamos a concatenar estos vectores con las características estilométricas para alimentar modelos de ML. A continuación, presentamos los resultados finales obtenidos con cada uno de los modelos de ML considerados.

| Subtarea  |                Modelo               | F1-Score |
|-----------|-------------------------------------|----------|
| Subtarea_1|        LogisticRegression (LR)      | 0.974962 |
| Subtarea_1| Extreme Gradient Boosting (XGBOOST) | 0.974380 |
| **Subtarea_1**|  **Stochastic Gradient Descent (SGD)**  | **0.975179** |
| Subtarea_1| Support Vector Classification (SVC) | 0.974808 |
| Subtarea_1|    Random Forest Classifier (RFC)   | 0.974850 |
| Subtarea_2|        LogisticRegression (LR)      | 0.87914  |
| Subtarea_2| Extreme Gradient Boosting (XGBOOST) | 0.87893  |
| Subtarea_2|  Stochastic Gradient Descent (SGD)  | 0.87499  |
| **Subtarea_2**| **Support Vector Classification (SVC)** | **0.88247**  |
| Subtarea_2|    Random Forest Classifier(RFC)    | 0.87382  |


# Funcionamiento ⚙️🔧
En esta sección se explica cómo se encuentran organizadas las distintas componentes del modelo, tanto para la primer subtarea como para la segunda, y cómo hacerlo funcionar para replicar los resultados obtenidos durantes los experimentos. Lo primero que se debe notar es que cada subtarea tiene asociada una carpeta en donde hemos colocado los siguientes archivos: 

- **Train_Test_S1.ipynb**/**Train_Test_S2.ipynb**:

Este notebook se utiliza para particionar de manera estratificada los datos en conjuntos de entrenamiento (70%) y prueba (30%). Es el primer
archivo que se necesita ejecutar para poder continuar con lo demás. 

- **LLM_S1.py**/**LLM_S2.py**:

Este script se utiliza para extraer los vectores embeddings de los modelos fine-tuneados. Ya que se tienen los conjuntos de entrenamiento y prueba, lo siguiente es ingresarlos a este script. Como resultado, se generará un documento por cada modelo que contenga, para cada texto, su vector embedding. Una observación importante es que los corpus resultantes no contienen la etiqueta que clasifica el texto en las disintas categorías de las subtareas. Sin embargo, mantienen el orden de los textos en los corpus originales, por lo que es trivial recuperarla. 

La forma de ejecutar este script es mediante la terminal. Requiere dos argumentos: la ruta del corpus y una variable que indique si se trata del conjunto de entrenamiento o prueba. A modo de ejemplo

python LLM_S1.py -i 'ruta_hacia_el_corpus/train_S1.csv' -v1 0 

El valor cero indica que se trata del conjunto de entrenamiento. Cualquier otro valor que se coloque en lugar del cero indicará que se trata del conjunto de prueba. 

- **Stylometry_Train_Test_S1.ipynb**/**Stylometry_Train_Test_S2.ipynb**:

A la par del script anterior, este notebook se ejecuta para extraer las características estilométricas del corpus de entrenamiento y prueba. Para su funcionamiento requiere descargarse la librería Stylometry, que se puede encontrar en el siguiente [enlace](https://github.com/jpotts18/stylometry).

- **modelos_finales_S1.py**/**modelos_finales_S2.py**:

Finalmente, es en este script donde concatenamos los vectores embeddings de cada modelo, las características estilométricas, y entrenamos un modelo de Machine Learning clásico. El código está diseñado para probar distintos modelos y distintas combinaciones, que van desde la concatenación de los vectores embeddings de dos modelos fine-tuneados, hasta la combinación de los tres junto con las características estilométricas, siendo este último el modelo final y definitivo con el que se participó. 

La forma de ejecutar este script es mediante la terminal. Requiere dos argumentos: el modelo y la concatenación de embeddings que se quiere probar. A modo de ejemplo

python modelos_finales_S1.py -v 'XGBOOST' -T 'sty_bert'

Los modelos que están incorporados son:

1. **XGBOOST**: Extreme Gradient Boosting
2. **LR**: LogisticRegression
3. **SVC**: Support Vector Classification
4. **SGD**: Stochastic Gradient Descent
5. **RFC**: Random Forest Classifier

Y las combinaciones que se pueden realizar son las siguientes:


 1. **sty_bert**: Características estilométricas con vector embedding del modelo BERT-Base-Multilingual
 2. **sty_e5**: Características estilométricas con vector embedding del modelo Multilingual-E5-Large
 3. **sty_rob**: Características estilométricas con vector embedding del modelo XLM-RoBERTa-Base
 4. **bert_e5**: Vectores embeddings de los modelos BERT-Base-Multilingual y Multilingual-E5-Large
 5. **bert_rob**: Vectores embeddings de los modelos BERT-Base-Multilingual y XLM-RoBERTa-Base
 6. **e5_roberta**: Vectores embeddings de los modelos Multilingual-E5-Large y XLM-RoBERTa-Base
 7. **sty_bert_e5**: Características estilométricas, y vectores embeddings de BERT-Base-Multilingual y Multilingual-E5-Large
 8. **sty_bert_rob**: Características estilométricas, y vectores embeddings de BERT-Base-Multilingual y XLM-RoBERTa-Base
 9. **sty_e5_rob**: Características estilométricas, y vectores embeddings de Multilingual-E5-Largel y XLM-RoBERTa-Base
 10. **bert_e5_rob**: Vectores embeddings de BERT-Base-Multilingual, Multilingual-E5-Large y XLM-RoBERTa-Base
 11. **sty_bert_e5_roberta**: Características estilométricas con vectores embeddings de BERT-Base-Multilingual, Multilingual-E5-Large y XLM-RoBERTa-Base

# Extras 📦

En la carpeta con nombre **Otros Experimentos** se podrán encontrar otras técnicas empleadas con el fin de aumentar la complejidad y el rendimeinto del modelo.  