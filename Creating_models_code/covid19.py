import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

##read training data header=0 because file has columns name at row 0
dataset=pd.read_csv(r"..\Dataset\Covid_Dataset.csv",header=0)

##converting from yes & no to binary (0,1)
encoder=LabelEncoder()
for (column_name, column_data) in dataset.iteritems():
  dataset[column_name]=encoder.fit_transform(column_data)

##drop columns with bad correlations
features_with_bad_correlation = ['Running Nose','Asthma','Chronic Lung Disease','Headache','Heart Disease','Diabetes','Fatigue','Gastrointestinal ','Wearing Masks','Sanitization from Market']
for feature in features_with_bad_correlation:
  dataset =dataset.drop(feature,axis=1)

##splite dataset into features and labels
dataset_features=dataset.copy()
dataset_labels=dataset_features.pop("COVID-19")

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
    layers.Dense(20,input_dim =10 ,activation='relu'),
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
model.fit(training_features, training_labels, epochs=10)

##Evaluate accuracy for test data
valdiation_loss, valdiation_acc, valdiation_precision, valdiation_recall = model.evaluate(valdiation_features,  valdiation_labels, verbose=2)
print('\nValdiation accuracy:', valdiation_acc)
print('Valdiation precision:', valdiation_precision)
print('Valdiation recall:', valdiation_recall)
print('Valdiation f1 scor:', (2*valdiation_precision*valdiation_recall)/( valdiation_precision+valdiation_recall))

##save the model as .h5 file
model.save(r'..\Models\covid19_model.h5')