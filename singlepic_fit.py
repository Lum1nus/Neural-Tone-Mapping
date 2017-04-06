import numpy as np
import keras
import imageio as iio
import colorsys as cs

np.random.seed(702)# =^_^=

#initial work with images
input_img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
true_img = iio.imread('./drago03/anyhere/dani_belgium.png')
rows, cols, depth = input_img.shape

#build luminance(lightness/brightness/yarkost?) map for hdr one
l_map = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        l_map[i,j] = 0.299*input_img.item(i,j,0) + 0.587*input_img.item(i,j,1) + 0.114*input_img.item(i,j,2)
        #its like conversion to YIQ anyway

#extract descriptors
descriptors = []
#whole_hist, bins = np.histogram(l_map.ravel(), 10, normed = True)
for i in range(rows):
    for j in range(cols//4, cols):
        if (i+j)%2==0:
            descriptor = np.array(input_img[i,j,:]) # 3 color channels of the pixel
            hist,bins = np.histogram(l_map[max(0,i-14):min(rows, i+14), max(cols//4, j-14):min(j+14, cols)].ravel(), 25, normed = True)
            descriptor = np.append(descriptor, hist)
            hist,bins = np.histogram(l_map[max(0,i-5):min(rows, i+5), max(cols//4, j-5):min(j+5, cols)].ravel(), 25, normed = True)
            descriptor = np.append(descriptor, hist)
            descriptors.append(descriptor) #add numpy array of features to the array of feature vectors
descriptors = np.array(descriptors)

#extract GT
GT = np.array([])
for i in range(rows):
    for j in range(cols//4, cols):
        if (i+j)%2==0:
            y, ii, qq = cs.rgb_to_yiq(true_img.item(i,j,0), true_img.item(i,j,1), true_img.item(i,j,2)) #its actually the same with hdr
            GT = np.append(GT, y)

#work with the model
model = keras.models.Sequential()
model.add(keras.layers.Dense(400, input_dim=53, activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(descriptors, GT, nb_epoch=100, batch_size=512)

model.save('./models/singlepic.h5')











