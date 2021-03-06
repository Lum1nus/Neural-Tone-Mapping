import numpy as np
import keras
import imageio as iio
import colorsys as cs
import cv2

np.random.seed(702)# =^_^=

#initial work with images
input_img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
true_img = cv2.imread('./drago03/anyhere/dani_belgium.png')
rows, cols, depth = input_img.shape
newrows = 128
newcols = 128*cols//rows

#build luminance(lightness/brightness/yarkost?) map for hdr one
in_lum = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        in_lum[i,j] = 0.299*input_img.item(i,j,2) + 0.587*input_img.item(i,j,1) + 0.114*input_img.item(i,j,0)
        #its like conversion to YIQ anyway
in_lum = cv2.resize(in_lum, (newcols,newrows), interpolation=cv2.INTER_AREA)
in_lum = np.reshape(in_lum, (newrows,newcols,1))
train_array = []
train_array.append(in_lum)
train_data = np.array(train_array)

#build luminance(lightness/brightness/yarkost?) map for hdr one
gt_lum = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        gt_lum[i,j] = 0.299*true_img.item(i,j,2) + 0.587*true_img.item(i,j,1) + 0.114*true_img.item(i,j,0)
        #its like conversion to YIQ anyway
gt_lum = cv2.resize(gt_lum, (newcols,newrows), interpolation=cv2.INTER_AREA)
gt_lum = np.reshape(gt_lum, (newrows,newcols,1))
gt_array = []
gt_array.append(gt_lum)
gt_data = np.array(gt_array)

#work with the model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, 8, 8, border_mode='same', activation='relu', input_shape=(newrows,newcols, 1)))
model.add(keras.layers.Conv2D(32, 8, 8, border_mode='same', activation='relu'))
model.add(keras.layers.Conv2D(1, 8, 8, border_mode='same', activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(train_data, gt_data, nb_epoch=1000)

model.save('./models/singlepic_conv.h5')












