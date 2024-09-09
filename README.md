# **IberAutextification 2024** 👩🏻 :robot: 
# Introducción 🔍
La nueva generación de los grandes modelos de lenguaje (LLMs) es capaz de generar textos fluidos, coherentes y plausibles en muchos idiomas, volviéndose difícil de diferenciar del texto escrito por humanos. Esto ha generado la preocupación de que los LLMs puedan ser utilizados con fines maliciosos, como la generación de noticias falsas, la difusión de opiniones polarizadas o el aumento de la credibilidad de campañas de phishing, entre otros. Estos ataques pueden realizarse en diferentes idiomas, dominios, modelos o estrategias de generación, lo que dificulta su moderación.

IberAuTexTification surge de la necesidad de desarrollar estrategias de moderación efectivas y generalizables para enfrentar estos modelos, cada vez más sofisticados. Es la segunda versión de la tarea compartida AuTextification en IberLEF 2023, ampliada a tres dimensiones: más modelos, más dominios y más idiomas de la Península Ibérica.

En IberAuTexTification 2024 se introducen dos sub-tareas. La primera consiste en una clasificación binaria con dos clases: identificar si un texto fue escrito por un humano o generado por un LLM. La segunda sub-tarea es una clasificación multiclase en la que se busca identificar qué modelo generó el texto producido por un LLM. 

# Metodología 💡 
## Corpus 📄📄
Para evitar que nuestros modelos presenten overfitting, realizamos una mezcla aleatoria del corpus de cada subtarea antes de dividirlo en conjuntos de entrenamiento y prueba. La división se realiza de manera estratificada, de modo que el 70% de los datos se asigna al conjunto de entrenamiento y el 30% restante al conjunto de validación, asegurando particiones bien equilibradas en ambos conjuntos. 

El conjunto de entrenamiento se encuentra [aqui](https://drive.google.com/drive/folders/1VdTmKAzrfFrL-MKEmsvEXjYKugrm5Rw7?usp=share_link)
Alli se puede almacenar cualquier cambio o procesamiento que se le haga al dataset

## Modelos 🧩🧩
El enfoque que nosotros adoptamos consiste en una arquitectura que incorpora Redes Neuronales de Grafos (GNN), Modelos Multilingües de Gran Escala (LLM) y características estilométricas. El diagrama general de la arquitectura de los modelos presentados en las subtareas compartidas se muestra a continuación 

![Descripción de la imagen](https://drive.google.com/uc?export=view&id=1Zzm_o999lkIjJ1NZNQ_8NeghzvQORxaI)

Este repositorio presenta la arquitectura que emplea los LLMs junto con las características estilométricas, una de las tres arquitecturas propuestas en las subtareas. En este modelo, primero se realiza un fine-tuning a tres modelos de gran escala (BERT-Base-Multilingual, Multilingual-E5-Large, XLM-RoBERTa-Base). Luego, una vez que los LLMs han sido ajustados, se extraen los vectores embeddings de la última capa de cada modelo. Esto se hace con el objetivo de capturar toda la información contextual contenida en estos vectores y concatenarla con características estilométricas extraídas directamente del corpus original. Finalmente, el vector resultante alimenta un modelo de machine learning tradicional: Stochastic Gradient Descent (SGD) para la primer subtarea y Support Vector Classification (SVC) para la segunda. 

# Resultados de los experimentos 

| Encabezado 1 | Encabezado 2 | Encabezado 3 |
|--------------|--------------|--------------|
| Celda 1      | Celda 2      | Celda 3      |
| Celda 4      | Celda 5      | Celda 6      |
| Celda 7      | Celda 8      | Celda 9      |

