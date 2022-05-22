import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import   MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

##read training data header=0 because file has columns name at row 0
dataset=pd.read_csv(r"..\Dataset\diabetes.csv",header=0)

##splite dataset into features and labels
dataset_features=dataset.copy()
dataset_labels=dataset_features.pop("Outcome")

##MinMax Scaling for feature that not scale to be in range 0 to 1
##for (feature_name, feature_data) in dataset_features.iteritems():
##    dataset_features[feature_name] = dataset_features[feature_name].astype('int64')
##    dataset_features[feature_name] = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset_features[[feature_name]])

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
    layers.Dense(20,input_dim = 8 ,activation='relu'),
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
model.fit(training_features, training_labels, epochs=50, batch_size = 1)

##Evaluate accuracy for test data
valdiation_loss, valdiation_acc, valdiation_precision, valdiation_recall = model.evaluate(valdiation_features,  valdiation_labels, verbose=2)
print('\nValdiation accuracy:', valdiation_acc)
print('Valdiation precision:', valdiation_precision)
print('Valdiation recall:', valdiation_recall)
print('Valdiation f1 scor:', (2*valdiation_precision*valdiation_recall)/( valdiation_precision+valdiation_recall))

#data0 = [[3,126,88,41,235,39.3,0.704,27],[5,88,66,21,23,24.4,0.342,30],[5,78,48,0,0,33.7,0.654,25]]
#data1 = [[7,196,90,0,0,39.8,0.451,41],[8,108,70,0,0,30.5,0.955,33],[8,120,0,0,0,30,0.183,38]]
#print('predict of data 0 = ',model.predict(data0))
#print('predict of data 1 = ',model.predict(data1))

##save the model as .h5 file
model.save(r'..\Models\diabetes_model.h5')