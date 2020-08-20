import numpy as np
import tools as tl
import consts
import cv2
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.models import Sequential


data = np.load('mnist.npz')


x_test = tl.reshaped_for_input(data['x_test'])/255
x_train = tl.reshaped_for_input(data['x_train'])/255
y_train = tl.OneHotEncoding(data['y_train'])
y_test = tl.OneHotEncoding(data['y_test'])



model = Sequential()

model.add(Conv2D(kernel_size = (3,3),filters = 8,activation = 'relu',input_shape = consts.input_shape))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(consts.classes,activation = consts.activation))

optimizer = SGD(lr = consts.lr,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer,loss = consts.loss)

## training the model
model.fit(x_train,y_train,epochs=consts.epochs,shuffle=True,validation_data=(x_test,y_test))

prediction = tl.FromOneHot(model.predict(x_test))
correct = tl.FromOneHot(y_test)
# print(np.equal(prediction,correct))

acc = str((np.sum(prediction == correct)/len(list(y_test))) * 100)
print('Acc :' + acc + '%')

model.save('mnist.h5')
