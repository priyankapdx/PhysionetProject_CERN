import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM , Flatten, LeakyReLU
from keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import ModelCheckpoints
from keras.layers import TimeDistributed
#from keras import backend as k
#from sklearn.metrics import r2_score
#Import the data
import scipy.io
import h5py 
from mat4py import loadmat
#save('all_patient_data_vital_signs.mat',-v7)
import keras_metrics
import keras.backend as K
from imblearn.over_sampling import SMOTE, SMOTENC

#def r2(y_true,y_pred):
#    SS_res=K.sum(K.square(y_true-y_pred))
#    SS_tot=K.sum(K.square(y_true-K.mean(y_true)))
#    return (1-SS_res/(SS_tot+K.epsilon()))
#####METRICS

from keras import backend as K

#Data Preprocessing
data = scipy.io.loadmat("all_data_try_1")
data=data['all_data_try_1']
data=np.array(data)
#print(data.shape)
np.random.shuffle(data)
np.random.shuffle(data)
np.random.shuffle(data)
print('This is the data shape:')
print(data.shape)
x_train=(data[0:,0:17,0:-1])
#print(data[0:20,0:,-1])
#x_validation=(data[3001:-1,0:,0:-1])

validation=scipy.io.loadmat("val_normalized_all_parameters")
validation=validation['val_normalized_all_parameters']
validation=np.array(validation)
print(validation[1:50,:,-1])
#np.save('validation',validation)
print('saved data')
#x_train=(x_train, (x_train.min(), x_train.max()), (0, 1))
#print(x_train)
#making y_train
[f,g,h]=data.shape
y_train_initial=np.array(data[0:,0:,-1])
[d,e]=y_train_initial.shape

#print(y_train_initial[0:20,:])
#print(y_train_initial.shape)
y_train_1=np.zeros(d)


for i in range(0,d):
        output_slice=y_train_initial[i,:]
        #print(output_slice_pred)
        slice_indices=np.flatnonzero(output_slice)
        #print(len(slice_indices))
        #print(slice_indices)
        if np.sum(slice_indices)>0:
            y_train_1[i]=1
        else:
            y_train_1[i]=0
#y_train=y_train_1[0:3000]
y_train=y_train_1
#y_validation=y_train_1[3001:-1]
#print(y_train[0:100])
#print(x_train.shape)
#print(y_train.shape)
#print(y_train[-5:-1,0:])
#print(x_train[1,0:,0:])


#Validation model
################################################################

validation_x=(validation[0:,0:17,0:-1])
[f,g,h]=validation.shape
y_val_initial=np.array(validation[0:,0:,-1])
[k,l]=y_val_initial.shape
print('hi')
#print(y_train_initial[0:20,:])
#print(y_train_initial.shape)
y_val_sepsis=np.zeros(k)
#y_val_index=np.zeros(d)

print('validation data size')
print(y_val_initial.shape)
for i in range(0,k):
        output_slice=y_val_initial[i,:]
        #print(output_slice_pred)
        slice_indices=np.flatnonzero(output_slice)
        #print(len(slice_indices))
        #print(slice_indices)
        if np.sum(slice_indices)>0:
            y_val_sepsis[i]=1
        else:
            y_val_sepsis[i]=0

# #print(y_val_index[0:100])
# print(y_val_sepsis[0:100])
     
from sklearn.utils import class_weight
print(x_train.shape)
print(y_train[:100])
#class_weight={0: 0.11, 1: 0.89}

#sm = SMOTE(random_state=27, ratio=1.0)
#x_train, y_train = sm.fit_sample(x_train, y_train)
model = Sequential()


model.add(Dense(256, batch_size=1, input_shape=(17,40), activation='tanh')) #x_train change to match what you want; paper used tanh activation
model.add(Dropout(0.2))

#model.add(Dense(256, activation='relu')) #x_train change to match what you want; paper used tanh activation
#model.add(Dropout(0.2))

#model.add(Dense(128, activation='relu')) 
#model.add(Dropout(0.2))

model.add(Dense(128))#, activation='tanh')) 
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add((Dense(64)))#, activation='tanh')))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add((Dense(64)))#, activation='tanh')))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add((Dense(32)))#, activation='tanh')))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add((Dense(32)))#, activation='tanh')))
model.add(LeakyReLU(alpha=0.3))

model.add(Flatten())

model.add((Dense(1, activation='sigmoid'))) #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.summary()
opt=keras.optimizers.Adam(lr=0.000001, decay=1e-6)


model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['acc']
)
weights = class_weight.compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)
print(weights)
history= model.fit(x_train,
    y_train,
    epochs=40,
    batch_size=1,
    validation_data=(validation_x,y_val_sepsis),
    class_weight=weights)

sepsis_count=0
sepsis_correct=0
no_sepsis_count=0
no_sepsis_correct=0
#y_actual=y_train
y_prediction=((model.predict(validation_x, batch_size=1)) > 0.5).astype(int)
for i in range(0,len(y_val_sepsis)):
    if y_val_sepsis[i]==1:
        sepsis_count=sepsis_count+1
        if y_prediction[i]==1:
            sepsis_correct=sepsis_correct+1
print(sepsis_count)
print(sepsis_correct)

for i in range(0,len(y_val_sepsis)):
    if y_val_sepsis[i]==0:
        no_sepsis_count=no_sepsis_count+1
        if y_prediction[i]==0:
            no_sepsis_correct=no_sepsis_correct+1
print(no_sepsis_count)
print(no_sepsis_correct)
#model.save('sepsis_flag.h5')
#score=model.evaluate(verbose=0)
#print('Test Loss:', score[0])
#print('Test Accuracy:', score[1])

model.save('sepsis_flag_working_40ep.h5')
#outputTensor = model.output
#listOfVariableTensors = model.trainable_weights
#gradients = k.gradients(outputTensor, listOfVariableTensors)

#print(gradients)
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["acc"], label="training accuracy")
plt.plot(history.history["val_loss"], label="validation loss")
plt.plot(history.history["val_acc"], label="validation accuracy")

#plt.plot(history.history["precision_m"], label="training precision")
#plt.plot(history.history["val_precision_m"], label="validation precision")
#plt.plot(history.history["recall_m"], label="training recall")
#plt.plot(history.history["val_recall_m"], label="validation recall")
#plt.plot(history.history["val_acc"], label="validation accuracy")
plt.legend()
plt.show()


#y_actual=y_train

#y_prediction_1=model.predict(x_train, batch_size=1)
#print(y_prediction_1)
#print(y_prediction.shape)
#print(y_prediction[0:10,0:42,0:])
#print(y_prediction_1.shape)
#print(y_prediction_1)


