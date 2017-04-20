import numpy as np
import keras
import imageio as iio
import colorsys as cs
import cv2

np.random.seed(702) # =^_^=

model = keras.models.load_model('./models/singlepic_conv.h5')
print 'Model is loaded'

img = iio.imread('./hdr/anyhere/dani_belgium.hdr', 'HDR-FI')
rows, cols, depth = img.shape
newrows = 128
newcols = 128*cols//rows

l_map = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        l_map.itemset((i,j), 0.299*img.item(i,j,0) + 0.587*img.item(i,j,1) + 0.114*img.item(i,j,2) )
sl_map = cv2.resize(l_map, (newcols,newrows), interpolation=cv2.INTER_AREA)
sl_map = np.reshape(sl_map, (newrows,newcols,1))
pred_array = []
pred_array.append(sl_map)
pred_data = np.array(pred_array)

predictions = model.predict(pred_data, verbose=1)

res_lum = predictions[0,:,:,0]

res_lum = cv2.resize(res_lum, (cols,rows))

alpha = 1.1		
for i in range(rows):
	for j in range(cols):
		r = img.item(i,j,0)
		g = img.item(i,j,1)
		b = img.item(i,j,2)
		img.itemset((i,j,2), (r/l_map.item(i,j))**alpha*res_lum.item(i,j) )
		img.itemset((i,j,1), (g/l_map.item(i,j))**alpha*res_lum.item(i,j) )
		img.itemset((i,j,0), (b/l_map.item(i,j))**alpha*res_lum.item(i,j) )

cv2.imwrite('./predicted/conv_2_32_8/dani_belgium.png', img)





