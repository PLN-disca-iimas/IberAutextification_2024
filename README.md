# **IberAutextification 2024**
Identificación de texto generado por IA

# Introducción 










## Datos
El conjunto de entrenamiento se encuentra [aqui](https://drive.google.com/drive/folders/1VdTmKAzrfFrL-MKEmsvEXjYKugrm5Rw7?usp=share_link)
Alli se puede almacenar cualquier cambio o procesamiento que se le haga al dataset

## Modelos 

En este repositorio se encuentran los experimentos realizados para la participación del equipo iimasNLP en las tareas compartidas de Identificación Automatizada de Textos en Lenguas de la Península Ibérica (IberAuTexTification 2024). 

La primer subtask consiste en determinar si un texto ha sido generado automáticamente o no. La segunda subtask consiste en clasificar qué modelo lo generó. Una novedad en esta edición es detectar sobre un entorno multilingüe, además de que se han añadido más dominios y modelos. 

En cada carpeta, para la subtask_1 y subtask_2, se encuentra un notebook (Train_test_S1.ipynb/Train_test_S2.ipynb) que muestra la forma en la que se particionaron los datos de entrenamiento (70%) y prueba (30%), un script (LLM_S1.py/LLM_S2.py) con el que se generaron los vectores embeddings de los tres modelos 'Fine-Tuneados' que se utilizaron y, finalmente, un script (modelos_finales_S1.py/modelos_finales_S2.py) que contiene todas las posibles configuraciones utilizadas para los experimentos, incluyendo la configuración que se utilizó para el modelo final.
