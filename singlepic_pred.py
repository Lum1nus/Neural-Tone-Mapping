import numpy as np
import keras
import imageio as iio
import colorsys as cs
import cv2

model = keras.models.load_model('./models/singlepic.h5')
print 'Model is loaded'

img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
rows, cols, depth = img.shape

#~ l_map = np.zeros((rows,cols))
#~ for i in range(rows):
    #~ for j in range(cols):
        #~ l_map[i,j] = 0.299*img.item(i,j,0) + 0.587*img.item(i,j,1) + 0.114*img.item(i,j,2)
        
#~ descriptors = []
#~ for i in range(rows):
    #~ for j in range(cols):
        #~ descriptor = np.array(img[i,j,:]) # 3 color channels of the pixel
        #~ hist,bins = np.histogram(l_map[max(0,i-14):min(rows, i+14), max(0, j-14):min(j+14, cols)].ravel(), 25, normed = True)
        #~ descriptor = np.append(descriptor, hist)
        #~ hist,bins = np.histogram(l_map[max(0,i-70):min(rows, i+70), max(0, j-70):min(j+70, cols)].ravel(), 25, normed = True)
        #~ descriptor = np.append(descriptor, hist)
        #~ descriptors.append(descriptor)
#~ descriptors = np.array(descriptors)

#~ np.save('./img_features/dani_belgium.npy', descriptors)
#~ print 'Features are saved'

descriptors = np.load('./img_features/dani_belgium.npy')

predictions = model.predict(descriptors, batch_size=256, verbose=1)

predictions = np.reshape(predictions, (rows,cols))

#convert to YIQ
for ii in range(rows):
    for jj in range(cols):
        y,i,q = cs.rgb_to_yiq(img.item(ii,jj,0), img.item(ii,jj,1), img.item(ii,jj,2))
        img.itemset((ii,jj,0), y)
        img.itemset((ii,jj,1), i)
        img.itemset((ii,jj,2), q)

img *=255
img[:,:,0] = predictions

#~ print img[100:105, 100:105, 0]
#~ print img[100:105, 100:105, 1]
#~ print img[100:105, 100:105, 2]

#convert back to BGR and save img
for ii in range(rows):
	for jj in range(cols):
		y = img.item(ii,jj,0)
		i = img.item(ii,jj,1)
		q = img.item(ii,jj,2)
		img.itemset((ii,jj,2), y + 0.956*i + 0.623*q)
		img.itemset((ii,jj,1), y - 0.272*i - 0.648*q)
		img.itemset((ii,jj,0), y - 1.105*i + 1.705*q)

cv2.imwrite('./predicted/dani_belgium.png', img)




