'''This code calculates the total volume removed below a user-defined reference plane
Written by Calvin Firth
Last updated Spring 2025'''

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from image_helper import open_image, partial_image
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
import numpy as np

path = str(input("Enter the path to your image: ")).strip(' "')

img, xycalibration, xy_unit, z_unit = open_image(path)

partial = str(input("Do you want to crop this image? [y/n]: "))
if(partial == "y"):
    img = partial_image(img)
    partial = True
elif(partial == "n"):
    partial = False
else:
    print("Invalid input. Answer \"y\" or \"n\".")

fig,ax = plt.subplots()
ax.imshow(img)
plt.title("Ablation Image", fontsize = 24)
plt.xlabel("x", fontsize = 22)
plt.ylabel("y", fontsize = 22)
plt.figtext(0.6, 0.5, "Click three points to define\nthe reference plane for depth calculations", fontsize = 17,backgroundcolor = 'white')
zoom_factory(ax)
ph = panhandler(fig,button=3)
points = clicker(ax,["click"],colors='k', disable_legend=True)
plt.show()

# User interface

if not points.get_positions()['click'].any():
    print("You didn't place any points!! Try Again.")
    quit()

x = np.arange(0, img.shape[1], 1)
x_og = xycalibration * x
y = np.arange(0, img.shape[0], 1)
y_og = xycalibration * y
x, y = np.meshgrid(x, y)


pt3 = points.get_positions()['click'][-1]
pt2 = points.get_positions()['click'][-2]
pt1 = points.get_positions()['click'][-3]

pt3 = np.append(pt3, [img[int(pt3[1])][int(pt3[0])]])
pt2 = np.append(pt2, [img[int(pt2[1])][int(pt2[0])]])
pt1 = np.append(pt1, [img[int(pt1[1])][int(pt1[0])]])

v1 = pt3-pt1
v2 = pt2-pt1
n = np.cross(v1,v2)

ref_plane = -((n[0]/n[2])*x + (n[1]/n[2])*y) + ((n[0]*pt1[0])/n[2] + ((n[1]*pt1[1])/n[2]) + pt1[2])
depth = ref_plane - img
volume = np.sum(depth)*xycalibration*xycalibration
print(f"Volume = {volume:.4f} ({z_unit}*{xy_unit}^2)")

x, y = np.meshgrid(x_og, y_og)

plt.figure()
plt.imshow(img)
plt.title("Original image")

plt.figure()
plt.imshow(depth)
plt.title("Depth below reference plane")
plt.show()