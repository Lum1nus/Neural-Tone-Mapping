import numpy as np
import keras
import imageio as iio
import cv2

model = keras.models.load_model('./models/singlepic.h5')
print 'Model is loaded'

img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
rows, cols, depth = img.shape

#l_map = np.zeros((rows,cols))
#for i in range(rows):
#    for j in range(cols):
#        l_map[i,j] = 0.299*img.item(i,j,0) + 0.587*img.item(i,j,1) + 0.114*img.item(i,j,2)
#        
#descriptors = []
#for i in range(rows):
#    for j in range(cols):
#        descriptor = np.array(img[i,j,:]) # 3 color channels of the pixel
#        hist,bins = np.histogram(l_map[max(0,i-14):min(rows, i+14), max(cols//4, j-14):min(j+14, cols)].ravel(), 10, normed = True)
#        descriptor = np.append(descriptor, hist)
#        hist,bins = np.histogram(l_map[max(0,i-70):min(rows, i+70), max(cols//4, j-70):min(j+70, cols)].ravel(), 10, normed = True)
#        descriptor = np.append(descriptor, hist)
#        descriptors.append(descriptor)
#descriptors = np.array(descriptors)

#np.save('./img_features/dani_belgium.npy', descriptors)
#print 'Features are saved'
descriptors = np.load('./img_features/dani_belgium.npy')

predictions = model.predict(descriptors, batch_size=256, verbose=1)

predictions = np.reshape(predictions, (rows,cols))

img[:,:,0] = predictions

img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
img = img*255
cv2.imwrite('./predicted/dani_belgium.png', img)




