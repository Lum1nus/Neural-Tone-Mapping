import numpy as np
import keras
import imageio as iio
import colorsys as cs
import cv2

np.random.seed(702) # =^_^=

model = keras.models.load_model('./models/singlepic.h5')
print 'Model is loaded'

img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
rows, cols, depth = img.shape

l_map = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        l_map.itemset((i,j), 0.299*img.item(i,j,0) + 0.587*img.item(i,j,1) + 0.114*img.item(i,j,2) )
        
#~ descriptors = []
#~ for i in range(rows):
    #~ for j in range(cols):
        #~ descriptor = np.array(img[i,j,:]) # 3 color channels of the pixel
        #~ hist,bins = np.histogram(l_map[max(0,i-14):min(rows, i+14), max(0, j-14):min(j+14, cols)].ravel(), 25, normed = True)
        #~ descriptor = np.append(descriptor, hist)
        #~ hist,bins = np.histogram(l_map[max(0,i-5):min(rows, i+5), max(0, j-5):min(j+5, cols)].ravel(), 25, normed = True)
        #~ descriptor = np.append(descriptor, hist)
        #~ descriptors.append(descriptor)
#~ descriptors = np.array(descriptors)

#~ np.save('./img_features/dani_belgium.npy', descriptors)
#~ print 'Features are saved'

descriptors = np.load('./img_features/dani_belgium.npy')

predictions = model.predict(descriptors, batch_size=512, verbose=1)

predictions = np.reshape(predictions, (rows,cols))

alpha = 1.1		
for i in range(rows):
	for j in range(cols):
		r = img.item(i,j,0)
		g = img.item(i,j,1)
		b = img.item(i,j,2)
		img.itemset((i,j,2), (r/l_map.item(i,j))**alpha*predictions.item(i,j) )
		img.itemset((i,j,1), (g/l_map.item(i,j))**alpha*predictions.item(i,j) )
		img.itemset((i,j,0), (b/l_map.item(i,j))**alpha*predictions.item(i,j) )

cv2.imwrite('./predicted/dani_belgium.png', img)





