import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

##read training data header=0 because file has columns name at row 0
dataset=pd.read_csv(r'..\Dataset\strokes.csv',header=0)

##fill all empty cell in bmi column with mean value
##And fill all empty cell in smoking_status column with Unknown value
dataset.bmi.fillna(dataset.bmi.mean(),inplace=True)
dataset.smoking_status.fillna('Unknown',inplace=True)

##count number of values for each class in gender and smoking_status columns 
columns_with_string_value = ['gender', 'smoking_status', 'ever_married']
for column_name in columns_with_string_value:
  print(dataset[column_name].value_counts(),'\n')

##replace Other value in gender column with Female
dataset["gender"] = dataset["gender"].replace(["Other"],"Female")

##give different value for each class in gender and smoking status columns 
encoder=LabelEncoder()
for column_name in columns_with_string_value:
  dataset[column_name]=encoder.fit_transform(dataset[column_name])

##drop columns with bad correlations
dataset = dataset.drop('work_type',axis=1)
dataset = dataset.drop('Residence_type',axis=1)

##splite dataset into features and labels
dataset_features=dataset.copy()
dataset_labels=dataset_features.pop("stroke")

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
    layers.Dense(20,input_dim =8 ,activation='relu'),
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

##save the model as .h5 file
model.save(r'..\Models\heart_stroke_model.h5')