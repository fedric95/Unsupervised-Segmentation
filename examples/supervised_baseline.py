from skimage import io
import numpy as np
import matplotlib.pyplot as plt

'''
    Note that after processing, the label will be in 0-6, where 
    0 - cropland 
    1 - forest 
    2 - grassland 
    3 - wetland 
    4 - water 
    5 - bare land 
    6 - others.
'''

igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255])

def load_lc(path):
    lc = io.imread(path)
    lc[lc == 255] = 12
    lc = igbp2hunan[lc]
    return lc



path = 'C:/Users/federico/Downloads/Hunan_Dataset/train/lc/lc_2604.tif'
label = load_lc(path)

path = 'C:/Users/federico/Downloads/Hunan_Dataset/train/s1/s1_2604.tif'
data = io.imread(path)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(data[:, :, 0])
axs[1].imshow(data[:, :, 1])
axs[2].imshow(label)
plt.show()