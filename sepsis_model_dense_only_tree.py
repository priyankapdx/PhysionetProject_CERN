import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM , Flatten, LeakyReLU
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
data = scipy.io.loadmat("normalized_all_data")
data=data['normalized']
data=np.array(data)
#print(data.shape)
np.random.shuffle(data)

validation=scipy.io.loadmat("validation_normalized")
validation=validation['normalized']
validation=np.array(validation)

#print(data.shape)
x_train=(data[0:,0:,0:-1])
#print(x_train[:9,:,:])
#x_train=x_train.reshape([0,4020,43,8])
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
            y_train_index[i]=index
        else:
            y_train_index[i]=g
        
        if np.sum(slice_indices)>0:
            y_train_sepsis[i]=1
        else:
            y_train_sepsis[i]=0

print(y_train_index[0:100])
print(y_train_sepsis[0:100])
#y_train=y_train_1[0:3000]
#y_validation=y_train_1[3001:-1]
#print(y_train[0:100])
#print(x_train.shape)
#print(y_train.shape)
#print(y_train[-5:-1,0:])
#print(x_train[1,0:,0:])

#Validation model
#################################################################

validation_x=(validation[0:,0:,0:-1])
[f,g,h]=validation.shape
y_val_initial=np.array(validation[0:,0:,-1])
[d,e]=y_val_initial.shape
print('hi')
#print(y_train_initial[0:20,:])
#print(y_train_initial.shape)
y_val_sepsis=np.zeros(d)
y_val_index=np.zeros(d)


for i in range(0,d):
        output_slice=y_train_initial[i,:]
        #print(output_slice_pred)
        slice_indices=np.flatnonzero(output_slice)
        #print(len(slice_indices))
        #print(slice_indices)
        if len(slice_indices)>0:
            index=slice_indices[0]
            #print(index)
            y_val_index[i]=index
        else:
            y_val_index[i]=g
        
        if np.sum(slice_indices)>0:
            y_val_sepsis[i]=1
        else:
            y_val_sepsis[i]=0

print(y_val_index[0:100])
print(y_val_sepsis[0:100])


########################
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
#print(len(x_train))
input_layer=Input(shape=(43,8))
#print(input_layer)
x= Dense(256, batch_size=1, input_shape=(43,8), activation='relu')(input_layer) #x_train change to match what you want; paper used tanh activation
x=Dropout(0.2)(x)


x=Dense(128)(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)

x=Dense(64)(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)

x=Dense(32)(x)
x=LeakyReLU(alpha=0.1)(x)
h=Flatten()(x)

dnn=Model(input_layer,h)

dnn_out=dnn(input_layer)


sepsis_flag=Dense(1, activation='sigmoid')(dnn_out) #model.add(TimeDistributed(Dense(1, activation='sigmoid')))
sepsis_index=Dense(1, activation='relu')(dnn_out)
model=Model(inputs=input_layer, outputs=[sepsis_flag, sepsis_index])
#return Model(input=input_shape, output=[sepsis_flag, sepsis_index])
model.summary()
opt=tf.keras.optimizers.Adam(lr=0.000001, decay=1e-6)

model.compile(
    optimizer=opt,
    loss=['binary_crossentropy','mean_squared_error'],
    metrics=['accuracy']
    #loss_weights=[0, 1]
)

history= model.fit(x_train,
    [y_train_sepsis,y_train_index],
    epochs=10,
    batch_size=1,
    validation_data=(validation_x,[y_val_sepsis,y_val_index])
    )
    #validation_data=(x_validation,y_validation))

output=model.predict(x_train, batch_size=1)
print(np.amax(output[0]))
print(np.amin(output[0]))
print(np.amax(y_train_sepsis))
print(np.amin(y_train_sepsis))
print(np.amax(y_train_index))
print(np.amin(y_train_index))
#score=sepsis_model.evaluate(x_validation,y_validation, verbose=0)
#print('Test Loss:', score[0])
#print('Test Accuracy:', score[1])


#outputTensor = model.output
#listOfVariableTensors = model.trainable_weights
#gradients = k.gradients(outputTensor, listOfVariableTensors)

#print(gradients)
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.show()

#y_actual=y_train
#y_prediction=((sepsis_model.predict(x_train, batch_size=1)) > 0.5).astype(int)
#y_prediction_1=sepsis_model.predict(x_train, batch_size=1)
#print(y_prediction.shape)
#print(y_prediction[0:10,0:42,0:])
#print(y_prediction_1.shape)
#print(y_prediction_1)
#sepsis_validation_function(y_actual,y_prediction)


with open("model.json", "w") as file:
    file.write(model.to_json())
model.save_weights("weights.h5")

from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

model_file = "model.json"
weights_file = "weights.h5"

with open(model_file, "r") as file:
    config = file.read()

K.set_learning_phase(0)
model = model_from_json(config)
model.load_weights(weights_file)

saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model")

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()