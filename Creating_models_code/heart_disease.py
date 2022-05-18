import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import   MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

##read training data header=0 because file has columns name at row 0
dataset=pd.read_csv(r"..\Dataset\heart_disease.csv",header=0)

##MinMax Scaling for feature that not scale to be in range 0 to 1
feature_to_scale = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
for feature in feature_to_scale: 
    dataset[feature] = dataset[feature].astype('int64')
    dataset[feature] = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset[[feature]])

##splite dataset into features and labels
dataset_features=dataset.copy()
dataset_labels=dataset_features.pop("HeartDiseaseorAttack")

##drop columns with bad correlations
features_with_bad_correlation = ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost']
for feature in features_with_bad_correlation: 
    dataset_features = dataset_features.drop(feature,axis=1)


##oversampling for dataset so that it will be balance(number of records for each class is equal) 
ros = RandomOverSampler()
dataset_features, dataset_labels = ros.fit_resample(dataset_features, dataset_labels)

##splite features and labels into training and valdiation
training_features,valdiation_features,training_labels,valdiation_labels = train_test_split(dataset_features,dataset_labels
                                                                               ,test_size=0.15,random_state=42)
##convert each row to array
training_features = np.array(training_features)
valdiation_features =  np.array(valdiation_features)

##setup model 
model = keras.Sequential([
    layers.Dense(300,input_dim =13 ,activation='relu'),
    layers.Dense(250,activation='relu'),
    layers.Dense(200,activation='relu'),
    layers.Dense(150,activation='relu'),
    layers.Dense(100,activation='relu'),
    layers.Dense(80,activation='relu'),
    layers.Dense(60,activation='relu'),
    layers.Dense(40,activation='relu'),
    layers.Dense(20,activation='relu'),
    layers.Dense(15,activation='relu'),
    layers.Dense(10,activation='relu'),
    layers.Dense(5,activation='relu'),
    layers.Dense(3,activation='relu'),
    layers.Dense(1,activation='sigmoid'),
])

##Compile the model
model.compile(loss = tf.keras.losses.MeanSquaredError(),
              optimizer = tf.optimizers.Adam(),
              metrics=['BinaryAccuracy','Precision', 'Recall'])

##trainign of the model
##model feeding ;epochs=number of iterations over training data 
model.fit(training_features, training_labels, epochs=50)

##Evaluate accuracy for test data
valdiation_loss, valdiation_acc, valdiation_precision, valdiation_recall = model.evaluate(valdiation_features,  valdiation_labels, verbose=2)
print('\nValdiation accuracy:', valdiation_acc)
print('Valdiation precision:', valdiation_precision)
print('Valdiation recall:', valdiation_recall)
print('Valdiation f1 scor:', (2*valdiation_precision*valdiation_recall)/( valdiation_precision+valdiation_recall))


data0 = [[0,0,1,26,1,0,0,3,0,15,0,0,7]]
data1 = [[1,1,1,37,1,1,2,5,0,0,1,1,10]]
print('predict of data 0 = ',model.predict(data0))
print('predict of data 1 = ',model.predict(data1))

model.save(r'..\Models\heart_disease_model.h5')