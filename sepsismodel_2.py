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

#Data Preprocessing
data = scipy.io.loadmat("augmented_normalized.mat")
data=data['augmented_normalized']
print(data.shape)
#x_train=data[0:,0:,0:-1]
#y_train=data[0:,0:,-1]
x_train=np.array(data[0:3000,0:,0:-1])
x_validation=np.array(data[3001:-1,0:,0:-1])
y_train=data[0:3000,0:,-1]
y_validation=data[3001:-1,0:,-1]

print(x_train.shape)
print(y_train.shape)
print(y_train[-5:-1,0:])
print(x_train[1,0:,0:])
#accuracy metric
import keras.backend as K
#Validation model
def sepsis_validation_function(y_actual, y_prediction):

    #y_prediction=np.random.randint(0,high=2, size=(5,7,1))
    #y_actual=np.random.randint(0,high=2, size=(5,7,1))
    #get size of two matrices
    [z,n,m]=y_prediction.shape
    [w,l]=y_actual.shape

    #check to see if the two matrices are the same shape
    #print(np.array_equal(y_prediction.shape,y_actual.shape))

    #initialize index matrix for y_prediction
    sepsis_count=0
    y_pred_indices=np.zeros(z)
    for i in range(0,z-1):
        output_slice_pred=y_prediction[i,:,:]
        #print(output_slice_pred)
        slice_indices_pred=np.flatnonzero(output_slice_pred)
        #print(slice_indices_pred)
        if len(slice_indices_pred)>0:
            index=slice_indices_pred[0]
            #print(index)
            y_pred_indices[i]=index
            sepsis_count=sepsis_count+1
        else:
            y_pred_indices[i]=1000
    #print(y_pred_indices)
    #initialize indices for actual matrix
    sepsis_count_act=0
    y_act_indices=np.zeros(w)
    for j in range(0,w-1):
        output_slice_act=y_actual[j,:]
        slice_indices_act=np.flatnonzero(output_slice_act)
        if len(slice_indices_act)>0:
            index=slice_indices_act[0]
            y_act_indices[j]=index
            sepsis_count_act=sepsis_count_act+1
        else:
            y_act_indices[j]= 300
    print(y_pred_indices.shape)
    print(y_act_indices.shape)

    print(np.array_equal(y_pred_indices.shape,y_act_indices.shape))



    t_diff=y_pred_indices - y_act_indices
    print(t_diff)
    accuracy=np.zeros(len(t_diff))

    for p in range (0,len(t_diff)):
    
        if t_diff[p] < -6:
            accuracy[p]=0
        elif t_diff[p] <= 0 and t_diff[p] >= -6:
            accuracy[p]=1/6*t_diff[p]+1
        elif t_diff[p] <= 7 and t_diff[p]> 0:
            accuracy[p]=-1/7*t_diff[p]+1
        elif t_diff[p] == 700:
            accuracy[p]=1
        elif t_diff[p]>7 and t_diff[p] != 700:
            accuracy[p]=0


    print(accuracy) 
    print(np.mean(accuracy))
    percent_sepsis_detected=(sepsis_count/sepsis_count_act)*100
    print(sepsis_count_act)
    print(percent_sepsis_detected)  

model = Sequential()


model.add(LSTM(43, batch_size=1, input_shape=(43,8), activation='tanh', return_sequences=True,kernel_initializer='random_uniform')) #x_train change to match what you want; paper used tanh activation
model.add(Dropout(0.2))

model.add(LSTM(43, activation='tanh', return_sequences=True)) 
model.add(Dropout(0.2))

model.add(LSTM(43, activation='tanh', return_sequences=True)) 
model.add(Dropout(0.2))

model.add(LSTM(43, activation='tanh', return_sequences=True)) 

model.add((Dense(1, activation='sigmoid'))) #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.summary()
opt=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)


model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.fit(x_train,
    y_train,
    epochs=5,
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
