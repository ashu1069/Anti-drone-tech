import cv2
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

#Image description in RGB
img = cv2.imread('C:\\Users\\user\\OneDrive\\Desktop\\images-71074957736\\4.png')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
img.shape

#Converting RGB to Grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(imgGray, cv2.COLOR_BGR2RGB))
plt.show()
imgGray.shape

#Getting the pixels into an array
import numpy as np
data=np.array(imgGray)

#Reducing the number of dimensions of array
flattened=data.flatten()

#edge detection and image pre-processing
mat_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])
mat_x = np.array([[ -1, 0, 1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])
  
filtered_image = cv2.filter2D(imgGray, -1, mat_y)
plt.imshow(filtered_image, cmap='gray')
filtered_image = cv2.filter2D(imgGray, -1, mat_x)
plt.imshow(filtered_image, cmap='gray')
