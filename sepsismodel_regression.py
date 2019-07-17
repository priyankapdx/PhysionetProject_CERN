import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM 
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import ModelCheckpoints
from tensorflow.keras.layers import TimeDistributed


#Import the data
import scipy.io
import h5py 
from mat4py import loadmat
#save('all_patient_data_vital_signs.mat',-v7)

import keras.backend as K

def custom_loss(y_true,y_pred):
    return K.mean(K.square(K.round(y_pred) - y_true), axis=-1)

#Data Preprocessing
data = scipy.io.loadmat("augmented_normalized")
data=data['augmented_normalized']
#print(data.shape)
x_train=np.array(data[0:3000,0:,0:-1])
x_validation=np.array(data[3001:-1,0:,0:-1])

#x_train=(x_train, (x_train.min(), x_train.max()), (0, 1))
print(x_train)
#making y_train
y_train_initial=np.array(data[0:,0:,-1])
[d,e]=y_train_initial.shape
#print(y_train_initial.shape)
y_train_1=np.zeros(d)


for i in range(0,d):
        output_slice=y_train_initial[i,:]
        #print(output_slice_pred)
        slice_indices=np.flatnonzero(output_slice)
        #print(slice_indices_pred)
        if len(slice_indices)>0:
            index=slice_indices[0]
            #print(index)
            y_train_1[i]=index
            
        else:
            y_train_1[i]=0
y_train=y_train_1[0:3000]
y_validation=y_train_1[3001:-1]

print(x_train.shape)
print(y_train.shape)
#print(y_train[-5:-1,0:])
#print(x_train[1,0:,0:])

#Validation model
def sepsis_validation_function(y_actual, y_prediction):

    t_diff=y_prediction - y_actual
    print(t_diff)
    accuracy=np.zeros(len(t_diff))

    for p in range (0,len(t_diff)):
    
        if t_diff[p] < -6:
            accuracy[p]=0
        elif t_diff[p] <= 0 and t_diff[p] >= -6:
            accuracy[p]=1/6*t_diff[p]+1
        elif t_diff[p] <= 7 and t_diff[p]> 0:
            accuracy[p]=-1/7*t_diff[p]+1
        elif t_diff[p] == 0:
            accuracy[p]=1
        elif t_diff[p]>7: #and t_diff[p] != 700:
            accuracy[p]=0


    print(accuracy) 
    print(np.mean(accuracy))
    #percent_sepsis_detected=(sepsis_count/sepsis_count_act)*100
    #print(sepsis_count_act)
    #print(percent_sepsis_detected)  

model = Sequential()


model.add(LSTM(43, batch_size=1, input_shape=(43,8), activation='tanh', return_sequences=True)) #x_train change to match what you want; paper used tanh activation
model.add(Dropout(0.2))

model.add(LSTM(43, activation='tanh', return_sequences=False)) 

model.add((Dense(1, activation='relu'))) #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.summary()
opt=tf.keras.optimizers.Adam(lr=0.000001, decay=1e-6)


model.compile(
    loss=custom_loss,
    optimizer=opt,
    metrics=['binary_accuracy']
)

model.fit(x_train,
    y_train,
    epochs=200,
    batch_size=1,
    validation_data=(x_validation,y_validation))

score=model.evaluate(x_validation,y_validation, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
y_actual=y_train
y_prediction=((model.predict(x_train, batch_size=1)) > 0.5).astype(int)
y_prediction_1=model.predict(x_train, batch_size=1)
print(y_prediction.shape)
#print(y_prediction[0:10,0:42,0:])
print(y_prediction_1.shape)
print(y_prediction_1)
sepsis_validation_function(y_actual,y_prediction)
