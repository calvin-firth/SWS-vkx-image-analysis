#Literally just opens a file and displays the image.

from image_helper import open_image
import matplotlib.pyplot as plt

path = str(input("Enter the path to your image: ")).strip(' "')

img, xycalibration, xy_unit, z_unit = open_image(path)

plt.imshow(img, extent = (0,img.shape[1]*xycalibration,img.shape[0]*xycalibration,0))
plt.show()