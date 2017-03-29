import numpy as np
import keras
import imageio as iio

model = keras.models.load_model('./models/singlepic.h5')
print 'Model is loaded'

img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
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

#convert to YIQ
for ii in range(rows):
    for jj in range(cols):
        y,i,q = cs.rgb_to_yiq(img.item(ii,jj,0), img.item(ii,jj,1), img.item(ii,jj,2))
        img.itemset((ii,jj,0), y)
        img.itemset((ii,jj,1), i)
        img.itemset((ii,jj,2), q)

img[:,:,0] = predictions

#convert back to RGB and save img
for ii in range(rows):
    for jj in range(cols):
        r,g,b = cs.yiq_to_rgb(img.item(ii,jj,0), img.item(ii,jj,1), img.item(ii,jj,2))
        img.itemset((ii,jj,0), r)
        img.itemset((ii,jj,1), g)
        img.itemset((ii,jj,2), b)

iio.imwrite('./predicted/dani_belgium.png', img)




