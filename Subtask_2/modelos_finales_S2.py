# Importamos las bibliotecas a usar
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
from sklearn.metrics import f1_score
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
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
    parser = argparse.ArgumentParser(description='Modelos finales S2.')
    parser.add_argument('-v1', help='variable1')
    parser.add_argument('-v2', help='variable2')
    parser.add_argument('-v3', help='variable3')
    parser.add_argument('-v4', help='variable4')
    parser.add_argument('-v5', help='variable5')
    args = parser.parse_args()
    print('*'*30)
    print('Comienza procesamiento de datos')

    ngram_range = eval(args.v1)
    analyzer=str(args.v2)
    min_df=int(args.v3)
    modelos=str(args.v4)
    classifier=str(args.v5)

    # Dataset con las características estilométricas
    train_stylometry=pd.read_csv('Train_finales/tylometry_train_S2.csv')
    test_stylometry=pd.read_csv('Test_finales/tylometry_test_S2.csv')
    train_stylometry = train_stylometry.values 
    test_stylometry=test_stylometry.values

    #Dataset LLM Bert
    train_bert=pd.read_csv('Train_finales/train_subtask2bert-base-multilingual-cased-finetuned-autext24-subtask2.csv',header=None)
    test_bert=pd.read_csv('Test_finales/test_subtask2bert-base-multilingual-cased-finetuned-autext24-subtask2.csv',header=None)
    train_bert=train_bert.values
    test_bert=test_bert.values

    # Dataset LLM Multilingual_e5 
    train_e5=pd.read_csv('Train_finales/train_subtask2multilingual-e5-large-finetuned-autext24-subtask2.csv', header=None)
    test_e5=pd.read_csv('Test_finales/test_subtask2multilingual-e5-large-finetuned-autext24-subtask2.csv',header=None)
    train_e5=train_e5.values
    test_e5=test_e5.values

    #Dataset roberta 
    train_roberta=pd.read_csv('Train_finales/train_subtask2xlm-roberta-base-finetuned-autext24-subtask2.csv',header=None)
    test_roberta=pd.read_csv('Test_finales/test_subtask2xlm-roberta-base-finetuned-autext24-subtask2.csv',header=None)
    train_roberta=train_roberta.values
    test_roberta=test_roberta.values


    # Dataset original
    train_data = pd.read_csv('Train/train_S2.csv')
    test_data = pd.read_csv('Test/test_S2.csv')
    etiquetas = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, clase in enumerate(etiquetas):
        train_data['label'] = np.where(train_data['label'] == clase, i, train_data['label'])

    for i, clase in enumerate(etiquetas):
        test_data['label'] = np.where(test_data['label'] == clase, i, test_data['label'])
    
    X_train_data=train_data['text']
    y_train_data=train_data['label']
    X_test_data=test_data['text']
    y_test_data=test_data['label']

    if modelos=='BERT+ME5+ROBERTA+STY+NGRAM':
        cv = CountVectorizer(ngram_range=ngram_range,analyzer=analyzer,min_df=min_df)
        
        # Concatenamos datos de entrenamiento 
        X_train_cv = cv.fit_transform(X_train_data)
        X_train_cv=add_feature1(X_train_cv,train_stylometry)
        X_train_cv=add_feature1(X_train_cv,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
        X_train_cv=add_feature1(X_train_cv,train_roberta)
    
        # Concatenamos datos de prueba
        X_test_cv=cv.transform(X_test_data)
        X_test_cv=add_feature1(X_test_cv,test_stylometry)
        X_test_cv=add_feature1(X_test_cv,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)
        X_test_cv=add_feature1(X_test_cv,test_roberta)
    
    elif modelos=='BERT+ME5+ROBERTA+STY':
        # Concatenamos datos de entrenamiento
        X_train_cv=add_feature1(train_stylometry,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
        X_train_cv=add_feature1(X_train_cv,train_roberta)
    
        # concatenamos datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)
        X_test_cv=add_feature1(X_test_cv,test_roberta)


    elif modelos=='BERT+ME5+STY+NGRAM':
        cv = CountVectorizer(ngram_range=ngram_range,analyzer=analyzer,min_df=min_df)
        
        # Concatenamos datos de entrenamiento 
        X_train_cv = cv.fit_transform(X_train_data)
        X_train_cv=add_feature1(X_train_cv,train_stylometry)
        X_train_cv=add_feature1(X_train_cv,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
    
        # Concatenamos datos de prueba
        X_test_cv=cv.transform(X_test_data)
        X_test_cv=add_feature1(X_test_cv,test_stylometry)
        X_test_cv=add_feature1(X_test_cv,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)

    elif modelos=='BERT+ME5+STY':
        # Concatenamos datos de entrenamiento
        X_train_cv=add_feature1(train_stylometry,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
    
        # concatenamos datos de prueba
        X_test_cv=add_feature1(test_stylometry,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)

    print('Termina procesamiento de datos')
    print('*'*30)

    # Calculamos más características estilométricas adicionales (Entrenamiento)
    num_digits= X_train_data.str.count('\d')
    num_stops = X_train_data.str.count('\s')    
    # Y las agregamos a nuestros datos 
    X_train_cv = add_feature2(X_train_cv, num_digits)
    X_train_cv = add_feature2(X_train_cv, num_stops)


    # Calculamos más características estilométricas adicionales (Prueba)
    num_digits_test= X_test_data.str.count('\d')
    num_stops_test = X_test_data.str.count('\s')
    # Y las agregamos a nuestros datos
    X_test_cv = add_feature2(X_test_cv, num_digits_test)
    X_test_cv = add_feature2(X_test_cv, num_stops_test)


    print('Comienza el entrenamiento')
    if classifier=='XGBOOST':
        modelo_xgb = xgb.XGBClassifier(random_state=0)
        modelo_xgb.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score 
        predictions = modelo_xgb.predict(X_test_cv)

    elif classifier=='LR':
        modelo_lr = LogisticRegression(random_state=0)
        modelo_lr.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score 
        predictions = modelo_lr.predict(X_test_cv)

    elif classifier=='SGD':
        modelo_SGD = SGDClassifier(random_state=0)
        modelo_SGD.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score 
        predictions = modelo_SGD.predict(X_test_cv)

    elif classifier=='SVC':
        modelo_svc = LinearSVC(random_state=0)
        modelo_svc.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score
        predictions = modelo_svc.predict(X_test_cv)
    
    elif classifier=='RFC':
        modelo_RFC = RandomForestClassifier(random_state=0)
        modelo_RFC.fit(X_train_cv, y_train_data)
        #Obtenemos las predicciones de nuestro modelo para el conjunto de prueba y calculamos el f1_score
        predictions = modelo_RFC.predict(X_test_cv)

    
    print('Finaliza entrenamiento')
    print('*'*30)

    predictions = predictions.astype(int)
    y_test_data = y_test_data.astype(int)


    score=f1_score(y_test_data,predictions, average='macro')
    print(score)

if __name__ == "__main__":
    main()
