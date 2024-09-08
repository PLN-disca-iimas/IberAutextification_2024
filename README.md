# **IberAutextification 2024** :robot: :woman_light_skin_tone:
# Introducción 
La nueva generación de los grandes modelos de lenguaje (LLMs) es capaz de generar textos fluidos, coherentes y plausibles en muchos idiomas, volviéndose difícil de diferenciar del texto escrito por humanos. Esto ha generado la preocupación de que los LLMs puedan ser utilizados con fines maliciosos, como la generación de noticias falsas, la difusión de opiniones polarizadas o el aumento de la credibilidad de campañas de phishing, entre otros. Estos ataques pueden realizarse en diferentes idiomas, dominios, modelos o estrategias de generación, lo que dificulta su moderación.

IberAuTexTification surge de la necesidad de desarrollar estrategias de moderación efectivas y generalizables para enfrentar estos modelos, cada vez más sofisticados. Es la segunda versión de la tarea compartida AuTextification en IberLEF 2023, ampliada a tres dimensiones: más modelos, más dominios y más idiomas de la Península Ibérica.

En IberAuTexTification 2024 se introducen dos sub-tareas. La primera consiste en una clasificación binaria con dos clases: identificar si un texto fue escrito por un humano o generado por un LLM. La segunda sub-tarea es una clasificación multiclase en la que se busca identificar qué modelo generó el texto producido por un LLM. 

# Metodología 
El enfoque que nosotros adoptamos consiste en una arquitectura que incorpora Redes Neuronales de Grafos (GNN), Modelos Multilingües de Gran Escala (LLM) y características estilométricas. 



## Datos
El conjunto de entrenamiento se encuentra [aqui](https://drive.google.com/drive/folders/1VdTmKAzrfFrL-MKEmsvEXjYKugrm5Rw7?usp=share_link)
Alli se puede almacenar cualquier cambio o procesamiento que se le haga al dataset

## Modelos 

En este repositorio se encuentran los experimentos realizados para la participación del equipo iimasNLP en las tareas compartidas de Identificación Automatizada de Textos en Lenguas de la Península Ibérica (IberAuTexTification 2024). 

La primer subtask consiste en determinar si un texto ha sido generado automáticamente o no. La segunda subtask consiste en clasificar qué modelo lo generó. Una novedad en esta edición es detectar sobre un entorno multilingüe, además de que se han añadido más dominios y modelos. 

En cada carpeta, para la subtask_1 y subtask_2, se encuentra un notebook (Train_test_S1.ipynb/Train_test_S2.ipynb) que muestra la forma en la que se particionaron los datos de entrenamiento (70%) y prueba (30%), un script (LLM_S1.py/LLM_S2.py) con el que se generaron los vectores embeddings de los tres modelos 'Fine-Tuneados' que se utilizaron y, finalmente, un script (modelos_finales_S1.py/modelos_finales_S2.py) que contiene todas las posibles configuraciones utilizadas para los experimentos, incluyendo la configuración que se utilizó para el modelo final.
