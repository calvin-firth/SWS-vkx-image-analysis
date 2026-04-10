'''This code calculates the cumulative logfluence I(x,y) for a single ablated line (not really)
as a function of scan parameters by summing all the Gaussian beams
This actually is the same code as the logfluence for pyramids, just adapted to make a pyramid with one vertical and one horizontal line
It also has the additional capability (compared to the notebook) of comparing the fluence to the depth from a real image
For details about the calculation, see the pdf in this directory
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
from scipy import ndimage
import pandas as pd

def log_fluence_func(x,y,gw,ls,Px,Py,SS,RR,n,I_0,a,phi_threshold=1): #The function that calculates the logfluence
    return n*(sum(np.where(I_0*np.exp(-2*(((x-(SS/RR)*i)**2 + (y-(j*ls + (Py-gw)/2))**2)/(a**2)))>phi_threshold,(np.log(I_0*np.exp(-2*(((x-(SS/RR)*i)**2 + (y-(j*ls + (Py-gw)/2))**2)/(a**2)))/phi_threshold)),0) for i in range(int(-(0.05*Px)/(SS/RR)),int((1.05*Px)/(SS/RR))+1) for j in range(int(gw/ls)+1)) + sum(np.where(I_0*np.exp(-2*(((x-(i*ls + (Px-gw)/2))**2 + (y-j*(SS/RR))**2)/a**2))>phi_threshold,np.log((I_0*np.exp(-2*(((x-(i*ls + (Px-gw)/2))**2 + (y-j*(SS/RR))**2)/a**2)))/phi_threshold),0) for i in range(int(gw/ls) + 1) for j in range(int(-(0.05*Py)/(SS/RR)),int((1.05*Py)/(SS/RR))+1)))

#Scan parameters
P = 30 # Power
gw =0 # Groove width (um)
ls = 20 # Line spacing (um)
Px = 100 # Pitch in the x direction
Py = 100 # Pitch in the y direction
SS = 300000 # Scan speed (um/s)
RR = 100000 # Rep rate (Hz)
n=1 # Number of layers
a = 14.76 # 1/e^2 beam radius
phi_threshold = 2 # J/cm^2

I_0 = 2*(P/RR)/(np.pi*(a*10**-4)**2)

read_image = False # This controls whether or not the calculated cumulative logfluence is compared to the depth profile of a real imaged pyramid
if (read_image):
    # If read_image is true, then you input an image with pyramids and then the code will calculate an "average unit cell"
    path = str(input("Enter the path to your image: ")).strip(' "')
    img, xycalibration, xy_unit, z_unit = open_image(path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Ablation Image", fontsize=24)
    plt.xlabel("x", fontsize=22)
    plt.ylabel("y", fontsize=22)
    plt.figtext(0.6, 0.5,
                "*OPEN FULL-SCREEN*\nLeft-click on the center of a pyramid tip\n\nLeft-click to place a point\nScroll to zoom\nRight-click and drag to pan\nRight-click to remove a point",
                fontsize=17, backgroundcolor='white')
    zoom_factory(ax)
    ph = panhandler(fig, button=3)
    points = clicker(ax, ["click"], colors='k', disable_legend=True)
    plt.show()

    # User interface

    if not points.get_positions()['click'].any():
        print("You didn't place any points!! Try Again.")
        quit()

    start = points.get_positions()['click'][-1]
    # get starting position
    x_i = start[0]
    y_i = start[1]

    px, py, px_unc, py_unc = get_pitch(img, xycalibration, xy_unit,target=0.5,plot=False)

    x = np.arange(0, img.shape[1], 1)
    x_og = xycalibration * x
    y = np.arange(0, img.shape[0], 1)
    y_og = xycalibration * y
    x, y = np.meshgrid(x_og, y_og)

    # More UI
    rect = plt.Rectangle((x_i, y_i), -(px / xycalibration), -(py / xycalibration), facecolor="none", edgecolor='k')
    print("For the following questions, refer to the figure if needed.")
    fig2, ax2 = plt.subplots()
    ax2.imshow(img)
    ax2.scatter(x_i, y_i, s=20, marker='*', c='k')
    box = ax2.add_patch(rect)
    plt.title("Ablation Image", fontsize=24)
    plt.xlabel("x", fontsize=22)
    plt.ylabel("y", fontsize=22)
    plt.legend([box], ["First Pyramid"])
    plt.show(block=False)
    plt.pause(0.2)
    rows = int(input("How many rows to stack? (Including start row, going downwards)"))
    cols = int(input("How many columns to stack? (Including start column, going rightwards)"))
    init = (x_i, y_i)

    row_arr = []
    for row in range(rows):
        row_arr.append(img[(int(init[1] + (row - 1) * (py / xycalibration))):(
            int(init[1] + (row - 1) * (py / xycalibration) + int(py / xycalibration)))])

    pyr_arr = []
    for row in row_arr:
        trn = np.transpose(row)
        for col in range(cols):
            # split image into sub-images, append them to pyr_arr
            pyr_arr.append(np.transpose(trn[(int(init[0] + (col - 1) * (px / xycalibration))):(
                int(init[0] + (col - 1) * (px / xycalibration) + int(py / xycalibration)))]))

    avg_pyramid = np.average(pyr_arr, axis=0)
    avg_pyr_unc = np.std(pyr_arr, axis=0)
    x_cut = avg_pyramid[avg_pyramid.shape[0] // 2]
    x_cut_unc = avg_pyr_unc[avg_pyramid.shape[0] // 2]
    y_cut = np.transpose(avg_pyramid)[avg_pyramid.shape[1] // 2]
    y_cut_unc = np.transpose(avg_pyr_unc)[avg_pyramid.shape[1] // 2]
    diagonal_cut = np.diagonal(avg_pyramid)
    diagonal_cut_unc = np.diagonal(avg_pyr_unc)

    depth = np.max(avg_pyramid) - avg_pyramid

    plt.figure()
    plt.imshow(depth, extent=(0, 1000 * px, 1000 * py, 0), cmap="viridis_r")
    plt.title("Average Unit Cell", fontsize=24)
    plt.xlabel("x (um)", fontsize=22)
    plt.ylabel("y (um)", fontsize=22)
    plt.colorbar(label="Depth (um)")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    '''plt.figure()
    plt.contour(x[0:avg_pyramid.shape[0], 0:avg_pyramid.shape[1]], y[0:avg_pyramid.shape[0], 0:avg_pyramid.shape[1]],
                avg_pyramid, colors='k')
    plt.contourf(x[0:avg_pyramid.shape[0], 0:avg_pyramid.shape[1]], y[0:avg_pyramid.shape[0], 0:avg_pyramid.shape[1]],
                 avg_pyramid)
    plt.title("Average Pyramid Contour Plot", fontsize=24)
    plt.xlabel("x (" + xy_unit + ")", fontsize=22)
    plt.ylabel("y (" + xy_unit + ")", fontsize=22)

    errbar(np.linspace(0, (xycalibration * x_cut.size), x_cut.size, endpoint=False), x_cut, x_cut_unc,
           "Average Pyramid x cross-section", "x (" + str(xy_unit) + ")", "z (" + str(z_unit) + ")")
    errbar(np.linspace(0, (xycalibration * y_cut.size), y_cut.size, endpoint=False), y_cut, y_cut_unc,
           "Average Pyramid y cross-section", "y (" + str(xy_unit) + ")", "z (" + str(z_unit) + ")")
    errbar(np.linspace(0, (np.sqrt(2) * xycalibration * diagonal_cut.size), diagonal_cut.size, endpoint=False),
           diagonal_cut, diagonal_cut_unc, "Average Pyramid diagonal cross-section", "y (" + str(xy_unit) + ")",
           "z (" + str(z_unit) + ")")'''
    x = np.linspace(0, 1000 * px, depth.shape[0])
    y = np.linspace(0, 1000 * py, depth.shape[1])

else:
    res = 2  # The resolution (in microns) of the lattice upon which you are evaluating the logfluence
    x = np.arange(0,Px,res)
    y = np.arange(0,Py,res)


x,y = np.meshgrid(x,y)
inv_f = (log_fluence_func(x,y,gw,ls,Px,Py,SS,RR,n,I_0,a,phi_threshold=phi_threshold))

plt.figure()
plt.imshow(inv_f, extent = (0,Px,Py,0),cmap="viridis_r")
plt.colorbar()
plt.title("Cumulative LogFluence", fontsize=24)
plt.xlabel("x (um)", fontsize = 22)
plt.ylabel("y (um)", fontsize =22)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.show()
