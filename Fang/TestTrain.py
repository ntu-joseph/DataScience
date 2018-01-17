from keras.models import Sequential
from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import Callback

import numpy as np
import pandas as pd
import datetime as dt
import h5py


import matplotlib.pyplot as plt

print("Import End")
"""
with h5py.File('tmp/train_tensor.h5', 'r') as hf:
    X = hf['features'][:]
    Y = hf['labels'][:]
"""
	
#讀取資料
test_train_X=pd.read_csv("./dataTemp.csv")
test_train_Y=pd.read_csv("./all_label.csv")

f_test=pd.read_csv("./dataTest.csv")

#print(test_train)
#print(test_data)

#切割測試資料
print("X")
test_train_X=test_train_X.iloc[:,0:33]
test_train_X=np.concatenate((test_train_X.iloc[:,0:2],test_train_X.iloc[:,3:33]),axis=1)
test_train_X=pd.DataFrame(test_train_X)
print(test_train_X.head())
print("Y")


f_test=f_test.iloc[:,0:33]
f_test=np.concatenate((f_test.iloc[:,0:2],f_test.iloc[:,3:33]),axis=1)
f_test=pd.DataFrame(f_test)
print(f_test,"F test")


print(test_train_Y.head())
print("X.SIZE")

print(test_train_X.size)
print("X CUT")
X_Array=test_train_X.as_matrix()
print(X_Array)
#print(X_Array.shape())
print("up X shape")
#X=X_Array.reshape(1,32,28,1)
#X=test_train_X.reshape(test_train_X[0],32,28,1)
X=test_train_X.values
#X=test_train_X.pivot_table(index=test_train_X[0],columns=test_train_X[:,1:31],values='USD', aggfunc=np.mean)
X=test_train_X.iloc[:,1:32]
X=X.values
print(X)


print("Test Shape")
print(X.shape)
print(test_train_Y.shape)
X=np.float32(X)
"""
for i in range(57159):
#    X[i][1]=float(X[i][1])/100
    for x in range(4,32):
        if(i<15):
           print("UP",X[i][x])
           print(X[i][1])
           print("FLOAT?",float(float(X[i][x])/float(X[i][1])))
        X[i][x]=float(float(X[i][x])/float(X[i][1]))
        if(i<15):
           print(X[i][x])
           print(X[i][1])
 #   X[i][1]=float(X[i][1])/10
   """ 

print("NEW X!",X)


#SPLIT
X=X.reshape(57159,1,31)
test_train_Y=test_train_Y.iloc[:,1:29]
print("AFTER CUT Y",test_train_Y)


Y=test_train_Y.values.reshape(57159,1,28)
print(Y.shape)
#Y=test_train_Y.values
print("RESHAPE?")
print(X)
print(X.shape)
print(Y.shape)
print(Y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)
print("Test Split")

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
"""
#CNN 1D
model = Sequential()
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=((10,32))))
model.add(Convolution1D(32, 3, border_mode='same'))
model.summary()
"""
"""
#CNN
model = Sequential()
#model.add(Conv2D(32, 3,, input_shape=(X_train.shape)))
model.add(Conv1D(32,3, activation='relu', padding='same',))
model.add(Dense(32, activation='relu', input_shape=(X_train.shape)))
#model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(28, activation='sigmoid'))

model.summary()
"""
#RNN

from keras.layers import LSTM
model = Sequential()
model.add(LSTM(units=200, return_sequences=True, input_shape=(1, 31)))
model.add(Dropout(0.375))
#model.add(LSTM(200, return_sequences=False))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.375))
model.add(Dense(28, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

from IPython.display import clear_output

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
"""
#CNN
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1
          #,callbacks=[plot_losses]
          ,validation_data=(X_test, y_test)
          )

score = model.evaluate(X_test, y_test)
print('test loss: {}'.format(score[0]))
print('test accuracy: {}'.format(score[1]))
"""
#RNN
model.fit(X_train, y_train, epochs=15, batch_size=256)

print("XY")
print(X_test)
print(y_test)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
f_test=f_test.iloc[:,1:32]
f_test=f_test.values.reshape(37092,1,31)
final_ans = model.predict(f_test)
##final_ans.
print("FA",final_ans)
print("FA SHAPE",final_ans.shape)
"""
import csv
s=csv.writer(final_ans,delimiter=',',lineterminator='\n')
s.writerow
final_ans
"""

a = np.asarray(final_ans.reshape(37092,28))
temp=a
np.savetxt("fooOrigin.csv", a, delimiter=",",fmt='%10.5f')
for i in range(37092):
    for x in range(28):
        if(a[i][x]>0.2):
            a[i][x]="1"
        if(a[i][x]<=0.2):
            a[i][x]="0"

#for i in range(37092):
#    a[i][0]=str(i+57159)

np.savetxt("foo20.csv", a, delimiter=",",fmt='%10.5f')
"""
a=temp
for i in range(37092):
    for x in range(28):
        if(a[i][x]>0.25):
            a[i][x]="1"
        if(a[i][x]<=0.25):
            a[i][x]="0"
np.savetxt("foo25.csv", a, delimiter=",",fmt='%10.5f')

a=temp

for i in range(37092):
    for x in range(28):
        if(a[i][x]>0.15):
            a[i][x]="1"
        if(a[i][x]<=0.15):
            a[i][x]="0"
np.savetxt("foo15.csv", a, delimiter=",",fmt='%10.5f')

a=temp
#float('%.6f'% a)
for i in range(37092):
    for x in range(28):
#        if(i<15):
 #           print(a[i][x])
        if(a[i][x]>=0.001):
            a[i][x]="1"
           # print("1")
        if(a[i][x]<0.001):
            a[i][x]="0"
print(a)
np.savetxt("fooWHAT.csv", a, delimiter=",",fmt='%10.5f')
"""

#final_ans.to_csv('predict.csv')

#np.savetxt("predict.txt", final_ans)
print()
print()
print()
print()
print()
print("Test End")