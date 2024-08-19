# Importamos las bibliotecas a usar
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import argparse

# Función para añadir más de una característica
def add_feature1(X, features_to_add):
    """
    Returns the sparse feature matrix X with an array 
    of features feature_to_add added.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(features_to_add)], 'csr')

# Función para añadir una característica a la vez
def add_feature2(X, feature_to_add):
    """
    Returns the sparse feature matrix X with a feature 
    feature_to_add added.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
        

def main():
    parser = argparse.ArgumentParser(description='Mejores modelos S2.')
    parser.add_argument('-v', help='Modelo')
    parser.add_argument('-T',help='Tipo de VE')
    args = parser.parse_args()
    modelo=str(args.v)
    ve=str(args.T)
    print('*'*30)
    print('Comienza procesamiento de datos')

    # Dataset con las características estilométricas
    train_stylometry=pd.read_csv('Train_finales/stylometry_train_S2.csv')
    test_stylometry=pd.read_csv('Test_finales/stylometry_test_S2.csv')
    train_stylometry = train_stylometry.values 
    test_stylometry=test_stylometry.values
    print('Stylometry listo!')

    #Dataset LLM Bert
    train_bert=pd.read_csv('Train_finales/train_subtask2bert-base-multilingual-cased-finetuned-autext24-subtask2.csv',header=None)
    test_bert=pd.read_csv('Test_finales/test_subtask2bert-base-multilingual-cased-finetuned-autext24-subtask2.csv',header=None)
    train_bert=train_bert.values
    test_bert=test_bert.values
    print('LLM Bert listo!')

    # Dataset LLM Multilingual_e5 
    train_e5=pd.read_csv('Train_finales/train_subtask2multilingual-e5-large-finetuned-autext24-subtask2.csv', header=None)
    test_e5=pd.read_csv('Test_finales/test_subtask2multilingual-e5-large-finetuned-autext24-subtask2.csv',header=None)
    train_e5=train_e5.values
    test_e5=test_e5.values
    print('LLM E5 listo!')

    #Dataset roberta 
    train_roberta=pd.read_csv('Train_finales/train_subtask2xlm-roberta-base-finetuned-autext24-subtask2.csv',header=None)
    test_roberta=pd.read_csv('Test_finales/test_subtask2xlm-roberta-base-finetuned-autext24-subtask2.csv',header=None)
    train_roberta=train_roberta.values
    test_roberta=test_roberta.values
    print('LLM Roberta listo!')


    # Dataset original
    train_data = pd.read_csv('Train_finales/train_S2.csv')
    test_data = pd.read_csv('Test_finales/test_S2.csv')
    etiquetas = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, clase in enumerate(etiquetas):
        train_data['label'] = np.where(train_data['label'] == clase, i, train_data['label'])

    for i, clase in enumerate(etiquetas):
        test_data['label'] = np.where(test_data['label'] == clase, i, test_data['label'])
    
    X_train_data=train_data['text']
    y_train_data=train_data['label']
    y_train_data=y_train_data.astype(int)

    X_test_data=test_data['text']
    y_test_data=test_data['label']
    y_test_data=y_test_data.astype(int)

    print('Datos originales, listos!')

    if ve=='sty_bert':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_stylometry,train_bert)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_stops)
        X_train_cv = add_feature2(X_train_cv, num_digits)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_bert)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')
    elif ve=='sty_e5':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_stylometry,train_e5)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_stops)
        X_train_cv = add_feature2(X_train_cv, num_digits)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_e5)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')
    elif ve=='sty_rob':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_stylometry,train_roberta)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_stops)
        X_train_cv = add_feature2(X_train_cv, num_digits)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_roberta)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')
    elif ve=='bert_e5':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_bert,train_e5)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_bert,test_e5)
        print('Datos de prueba listos!')
    elif ve=='bert_rob':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_bert,train_roberta)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_bert,test_roberta)
        print('Datos de prueba listos!')
    elif ve=='e5_roberta':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_e5,train_roberta)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_e5,test_roberta)
        print('Datos de prueba listos!')
    elif ve=='sty_bert_e5':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_stylometry,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_stops)
        X_train_cv = add_feature2(X_train_cv, num_digits)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')
    elif ve=='sty_bert_rob':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_stylometry,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_roberta)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_stops)
        X_train_cv = add_feature2(X_train_cv, num_digits)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_roberta)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')
    elif ve=='sty_e5_rob':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_stylometry,train_e5)
        X_train_cv=add_feature1(X_train_cv,train_roberta)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_stops)
        X_train_cv = add_feature2(X_train_cv, num_digits)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_e5)
        X_test_cv=add_feature1(X_test_cv,test_roberta)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')
    elif ve=='bert_e5_rob':
        # Concatenamos los datos de entrenamiento  
        X_train_cv=add_feature1(train_bert,train_e5)
        X_train_cv=add_feature1(X_train_cv,train_roberta)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_bert,test_e5)
        X_test_cv=add_feature1(X_test_cv,test_roberta)
        print('Datos de prueba listos!')
    elif ve=='sty_bert_e5_roberta':
        # Concatenamos los datos de entrenamiento 
        X_train_cv=add_feature1(train_stylometry,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
        X_train_cv=add_feature1(X_train_cv,train_roberta)
        # Calculamos más características estilométricas adicionales 
        num_digits= X_train_data.str.count(r'\d')
        num_stops = X_train_data.str.count(r'\s')    
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_digits)
        X_train_cv = add_feature2(X_train_cv, num_stops)
        print('Datos de entrenamiento listos!')
        # Concatenamos los datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)
        X_test_cv=add_feature1(X_test_cv,test_roberta)
        # Calculamos más características estilométricas adicionales
        num_digits_test= X_test_data.str.count(r'\d')
        num_stops_test = X_test_data.str.count(r'\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Datos de prueba listos!')

    print('Termina procesamiento de datos')
    print('*'*30)
    #Definimos el modelo a usar y lo entrenamos  
    print('Comienza el entrenamiento')
    if modelo=='XGBOOST':
        modelo_xgb = xgb.XGBClassifier(random_state=0)
        modelo_xgb.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score 
        predictions = modelo_xgb.predict(X_test_cv)

    elif modelo=='LR':
        modelo_lr = LogisticRegression(random_state=0)
        modelo_lr.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score 
        predictions = modelo_lr.predict(X_test_cv)

    elif modelo=='SGD':
        modelo_SGD = SGDClassifier(random_state=0)
        modelo_SGD.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score 
        predictions = modelo_SGD.predict(X_test_cv)

    elif modelo=='SVC':
        modelo_svc = LinearSVC(random_state=0)
        modelo_svc.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score
        predictions = modelo_svc.predict(X_test_cv)
    
    elif modelo=='RFC':
        modelo_RFC = RandomForestClassifier(random_state=0)
        modelo_RFC.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score
        predictions = modelo_RFC.predict(X_test_cv)

    print('Finaliza entrenamiento')
    print('*'*30)

    predictions = predictions.astype(int)


    score=f1_score(y_test_data,predictions, average='macro')
    print(score)

if __name__ == "__main__":
    main()
