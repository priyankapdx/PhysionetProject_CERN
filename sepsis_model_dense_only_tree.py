import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM , Flatten
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import ModelCheckpoints
from tensorflow.keras.layers import TimeDistributed
from keras import backend as k
from tensorflow.keras.layers import Input
#from sklearn.metrics import r2_score
#Import the data
import scipy.io
import h5py 
from mat4py import loadmat
#save('all_patient_data_vital_signs.mat',-v7)

import keras.backend as K

#def r2(y_true,y_pred):
#    SS_res=K.sum(K.square(y_true-y_pred))
#    SS_tot=K.sum(K.square(y_true-K.mean(y_true)))
#    return (1-SS_res/(SS_tot+K.epsilon()))

#Data Preprocessing
data = scipy.io.loadmat("missing_data_filled")
data=data['missing_data_filled']
data=np.array(data)
#print(data.shape)
np.random.shuffle(data)

#print(data.shape)
x_train=(data[0:,0:,0:-1])
x_train=x_train.reshape([-1,4020, 43,8])
print(x_train.shape)
#print(data[0:20,0:,-1])
#x_validation=(data[3001:-1,0:,0:-1])

#x_train=(x_train, (x_train.min(), x_train.max()), (0, 1))
#print(x_train)
#making y_train
[f,g,h]=data.shape
y_train_initial=np.array(data[0:,0:,-1])
[d,e]=y_train_initial.shape
print('hi')
#print(y_train_initial[0:20,:])
#print(y_train_initial.shape)
y_train_sepsis=np.zeros(d)
y_train_index=np.zeros(d)


for i in range(0,d):
        output_slice=y_train_initial[i,:]
        #print(output_slice_pred)
        slice_indices=np.flatnonzero(output_slice)
        #print(len(slice_indices))
        #print(slice_indices)
        if len(slice_indices)>0:
            index=slice_indices[0]
            #print(index)
            y_train_sepsis[i]=index
        else:
            y_train_sepsis[i]=g
        
        if np.sum(slice_indices)>0:
            y_train_index[i]=1
        else:
            y_train_index[i]=0


#y_train=y_train_1[0:3000]
#y_validation=y_train_1[3001:-1]
#print(y_train[0:100])
#print(x_train.shape)
#print(y_train.shape)
#print(y_train[-5:-1,0:])
#print(x_train[1,0:,0:])

#Validation model
def sepsis_validation_function(y_actual, y_prediction):

    t_diff=y_prediction[0:,0] - y_actual
    #print(y_prediction.shape)
    #print(y_actual.shape)
    #print(t_diff.shape)
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

#model = Model()
print(x_train.shape)
input_layer=Input(shape=(4020,43,8))
print(input_layer)
x= Dense(256, batch_size=1, input_shape=(43,8), activation='relu')(input_layer) #x_train change to match what you want; paper used tanh activation
x=Dropout(0.2)(x)


x=Dense(128, activation='relu')(x)
x=Dropout(0.2)(x)

x=Dense(64, activation='relu')(x)
x=Dropout(0.2)(x)

x=Dense(32, activation='relu')(x)
h=Flatten()(x)

dnn=Model(input_layer,h)

dnn_out=dnn(input_layer)


sepsis_flag=Dense(1, activation='sigmoid')(dnn_out) #model.add(TimeDistributed(Dense(1, activation='sigmoid')))
sepsis_index=Dense(1, activation='linear')(dnn_out)
sepsis_model=Model(inputs=input_layer, outputs=[sepsis_flag, sepsis_index])
#return Model(input=input_shape, output=[sepsis_flag, sepsis_index])
sepsis_model.summary()
opt=tf.keras.optimizers.Adam(lr=0.000001, decay=1e-6)

sepsis_model.compile(
    optimizer=opt,
    loss=['binary_crossentropy','mean_squared_error'],
    #loss_weights=[]
)

history= sepsis_model.fit(x_train,
    [y_train_sepsis,y_train_index],
    [sepsis_flag,sepsis_index],
    epochs=100)
    #batch_size=1)
    #validation_data=(x_validation,y_validation))

#score=sepsis_model.evaluate(x_validation,y_validation, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


#outputTensor = model.output
#listOfVariableTensors = model.trainable_weights
#gradients = k.gradients(outputTensor, listOfVariableTensors)

#print(gradients)
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()

y_actual=y_train
y_prediction=((sepsis_model.predict(x_train, batch_size=1)) > 0.5).astype(int)
y_prediction_1=sepsis_model.predict(x_train, batch_size=1)
#print(y_prediction.shape)
#print(y_prediction[0:10,0:42,0:])
#print(y_prediction_1.shape)
#print(y_prediction_1)
sepsis_validation_function(y_actual,y_prediction)
