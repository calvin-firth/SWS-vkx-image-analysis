'''This code calculates the average pyramid from an image and exports it as a csv
Written by Calvin Firth (UMN)
Last updated Fall 2025'''

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
from plotting import errbar
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
from image_helper import get_pitch, open_image
from get_parameters import get_params

path = str(input("Enter the path to your image: ")).strip(' "')

img, xycalibration, xy_unit, z_unit = open_image(path)
#print(xycalibration)
#print(img.shape)
#img = img[int(img.shape[0]*.1):int(img.shape[0]*.9),int(img.shape[1]*.1):int(img.shape[1]*.9)]
#print(img.shape)
fig,ax = plt.subplots()
ax.imshow(img)
plt.title("Ablation Image", fontsize = 24)
plt.xlabel("x", fontsize = 22)
plt.ylabel("y", fontsize = 22)
plt.figtext(0.6, 0.5, "*OPEN FULL-SCREEN*\nLeft-click on the bottom right trench\nof the first pyramid to be included\n(first meaning top left of pyramids included)\n\nLeft-click to place a point\nScroll to zoom\nRight-click and drag to pan\nRight-click to remove a point", fontsize = 17,backgroundcolor = 'white')
zoom_factory(ax)
ph = panhandler(fig,button=3)
points = clicker(ax,["click"],colors='k', disable_legend=True)
plt.show()

# User interface

if not points.get_positions()['click'].any():
    print("You didn't place any points!! Try Again.")
    quit()

start = points.get_positions()['click'][-1]
# get starting position
x_i = start[0]
y_i = start[1]

print("The following pitch values are calculated based on the *entire* input image: ")
px,py,px_unc,py_unc = get_pitch(img,xycalibration,xy_unit)


x = np.arange(0, img.shape[1], 1)
x_og = xycalibration * x
y = np.arange(0, img.shape[0], 1)
y_og = xycalibration * y
x, y = np.meshgrid(x_og, y_og)

# More UI
rect = plt.Rectangle((x_i,y_i),-(px/xycalibration),-(py/xycalibration),facecolor="none",edgecolor='k')
print("For the following questions, refer to the figure if needed.")
fig2,ax2 = plt.subplots()
ax2.imshow(img)
ax2.scatter(x_i,y_i, s=20,marker='*',c='k')
box = ax2.add_patch(rect)
plt.title("Ablation Image", fontsize = 24)
plt.xlabel("x", fontsize = 22)
plt.ylabel("y", fontsize = 22)
plt.legend([box],["First Pyramid"])
plt.show(block=False)
plt.pause(0.2)
rows = int(input("How many rows to stack? (Including start row, going downwards)"))
cols = int(input("How many columns to stack? (Including start column, going rightwards)"))
init = (x_i,y_i)

row_arr = []
for row in range(rows):
    row_arr.append(img[(int(init[1] + (row - 1)*(py/xycalibration))):(int(init[1] + (row - 1)*(py/xycalibration) + int(py/xycalibration)))])

pyr_arr = []
for row in row_arr:
    trn = np.transpose(row)
    for col in range(cols):
        # split image into sub-images, append them to pyr_arr
        pyr_arr.append(np.transpose(trn[(int(init[0] + (col-1)*(px/xycalibration))):(int(init[0] + (col-1)*(px/xycalibration) + int(py/xycalibration)))]))
n=0
avg_params = []
for pyr in pyr_arr:
    # find parameters for each pyramid
    avg_params.append(get_params(pyr, x_og, y_og,plot=False))
    n += 1

avg_pyramid = np.average(pyr_arr,axis=0)
avg_pyr_unc = np.std(pyr_arr, axis = 0)
x_cut = avg_pyramid[avg_pyramid.shape[0] // 2]
x_cut_unc = avg_pyr_unc[avg_pyramid.shape[0] // 2]
y_cut = np.transpose(avg_pyramid)[avg_pyramid.shape[1] // 2]
y_cut_unc = np.transpose(avg_pyr_unc)[avg_pyramid.shape[1] // 2]
diagonal_cut = np.diagonal(avg_pyramid)
diagonal_cut_unc = np.diagonal(avg_pyr_unc)

plt.figure()
plt.imshow(avg_pyramid, interpolation='nearest')
plt.title("Average Pyramid", fontsize = 24)

plt.figure()
plt.contour(x[0:avg_pyramid.shape[0],0:avg_pyramid.shape[1]],y[0:avg_pyramid.shape[0],0:avg_pyramid.shape[1]], avg_pyramid,colors = 'k')
plt.contourf(x[0:avg_pyramid.shape[0],0:avg_pyramid.shape[1]],y[0:avg_pyramid.shape[0],0:avg_pyramid.shape[1]], avg_pyramid)
plt.title("Average Pyramid Contour Plot", fontsize = 24)
plt.xlabel("x (" + xy_unit + ")", fontsize = 22)
plt.ylabel("y (" + xy_unit + ")", fontsize =22)

parameters = get_params(avg_pyramid, x_og, y_og, plot = False)

x = np.arange(-4, avg_pyramid.shape[1]+4, 1)
x_og = xycalibration * x
y = np.arange(-4, avg_pyramid.shape[0]+4, 1)
y_og = xycalibration * y
x,y = np.meshgrid(x_og,y_og)

x_csv = np.array((x*1000)[::2,::2])
y_csv = np.array((y*1000)[::2,::2])
z_csv = np.array((avg_pyramid*1000)[::2,::2]) # The number after the two colons (ie ::n) decreases the density of points along that axis by a factor of n
pad_val = np.min(z_csv)
z_csv = np.pad(z_csv,(2,),constant_values = pad_val) # Ring of "zeros" around the outer edge

plt.figure()
plt.imshow(z_csv)
plt.show()

x_csv = x_csv.flatten()
y_csv = y_csv.flatten()
z_csv = z_csv.flatten()

export = np.transpose(np.array([x_csv,y_csv,z_csv])) # This prepares an export format of x|y|z columns

#np.savetxt("test.csv", export,delimiter = ',') # Save the csv