'''The code that you run to actually get the parameters and the average pyramid
You enter the filepath to the image you want to analyze
Once the analysis is done, the parameters are printed out and the average pyramid is displayed
Written by Calvin Firth (UMN)
Last updated Fall 2025'''

import tabulate
import matplotlib.pyplot as plt
import numpy as np
from plotting import errbar
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
from image_helper import get_pitch, open_image, partial_image
from get_parameters import get_params

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

offset = str(input("Do you want the code to calculate the offset? [y/n]: "))
if(offset == "y"):
    offset = True
elif(offset == "n"):
    offset = False
else:
    print("Invalid input. Answer \"y\" or \"n\".")

if offset:
    fig,ax = plt.subplots()
    ax.imshow(img)
    plt.title("Ablation Image", fontsize = 24)
    plt.xlabel("x", fontsize = 22)
    plt.ylabel("y", fontsize = 22)
    plt.figtext(0.6, 0.5, "Click three points to define the\nreference plane for offset measurements", fontsize = 17,backgroundcolor = 'white')
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

if(partial):
    print("\nThe following pitch values are calculated based only on the cropped part of the input image: ")

else:
    print("\nThe following pitch values are calculated based on the *entire* input image: ")

z_peak = False
px,py,px_unc,py_unc = get_pitch(img,xycalibration,xy_unit,z_peak,300)

x = np.arange(0, img.shape[1], 1)
x_og = xycalibration * x
y = np.arange(0, img.shape[0], 1)
y_og = xycalibration * y
x, y = np.meshgrid(x_og, y_og)

# More UI
rect = plt.Rectangle((x_i,y_i),-(px/xycalibration),-(py/xycalibration),facecolor="none",edgecolor='k')
print("\nFor the following questions, refer to the figure if needed.")
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
print("")
init = (x_i,y_i)

row_arr = []
plane_row = []
for row in range(rows):
    row_arr.append(img[(int(init[1] + (row - 1)*(py/xycalibration))):(int(init[1] + (row - 1)*(py/xycalibration) + int(py/xycalibration)))])
    if offset:
        plane_row.append(ref_plane[(int(init[1] + (row - 1)*(py/xycalibration))):(int(init[1] + (row - 1)*(py/xycalibration) + int(py/xycalibration)))])

pyr_arr = []
plane_vals = []
for row in row_arr:
    trn = np.transpose(row)
    trn_plane = np.transpose(plane_row)
    for col in range(cols):
        # split image into sub-images, append them to pyr_arr
        pyr_arr.append(np.transpose(trn[(int(init[0] + (col-1)*(px/xycalibration))):(int(init[0] + (col-1)*(px/xycalibration) + int(py/xycalibration)))]))
        if offset:
            plane_vals.append(np.median(np.transpose(trn_plane[(int(init[0] + (col-1)*(px/xycalibration))):(int(init[0] + (col-1)*(px/xycalibration) + int(py/xycalibration)))])))

avg_params = []
num_pyr = 0
for pyr in pyr_arr:
    # find parameters for each pyramid
    avg_params.append(get_params(pyr, x_og, y_og,offset = offset, plane = plane_vals, num_pyr = num_pyr,plot=False))
    num_pyr += 1

avg_pyramid = np.average(pyr_arr,axis=0)
avg_pyr_unc = np.std(pyr_arr, axis = 0)
x_cut = avg_pyramid[avg_pyramid.shape[0] // 2]
x_cut_unc = avg_pyr_unc[avg_pyramid.shape[0] // 2]
y_cut = np.transpose(avg_pyramid)[avg_pyramid.shape[1] // 2]
y_cut_unc = np.transpose(avg_pyr_unc)[avg_pyramid.shape[1] // 2]
diagonal_cut = np.diagonal(avg_pyramid)
diagonal_cut_unc = np.diagonal(avg_pyr_unc)

plt.figure()
print(avg_pyramid.shape[0])
print(x_og[avg_pyramid.shape[0]])
plt.imshow(avg_pyramid, extent = (0,x_og[avg_pyramid.shape[0]],y_og[avg_pyramid.shape[1]],0),interpolation='nearest')
plt.title("Average Pyramid", fontsize = 24)
plt.xlabel("x (" + xy_unit + ")", fontsize = 22)
plt.ylabel("y (" + xy_unit + ")", fontsize =22)

plt.figure()
plt.contour(avg_pyramid,extent = (0,x_og[avg_pyramid.shape[0]],y_og[avg_pyramid.shape[1]],0),colors = 'k',origin='upper')
plt.contourf(avg_pyramid,extent = (0,x_og[avg_pyramid.shape[0]],y_og[avg_pyramid.shape[1]],0),origin='upper')
plt.gca().invert_yaxis()
plt.title("Average Pyramid Contour Plot", fontsize = 24)
plt.xlabel("x (" + xy_unit + ")", fontsize = 22)
plt.ylabel("y (" + xy_unit + ")", fontsize =22)

parameters = get_params(avg_pyramid, x_og, y_og, plot = False)

print("\nParameters of the average pyramid: ")
print(f"dx: {parameters[0]:.3f} {z_unit}")
print(f"dy: {parameters[1]:.3f} {z_unit}")
print(f"dt: {parameters[2]:.3f} {z_unit}")
print(f"average height: {parameters[3]:.3f} {z_unit}")
print(f"wx: {parameters[4]:.3f} {xy_unit}")
print(f"wy: {parameters[5]:.3f} {xy_unit}")

params = np.nanmean(avg_params, axis = 0)
param_unc = np.nanstd(avg_params, axis = 0)


def format_val_unc(val, unc, sigfigs=2):
    """Format value ± uncertainty with given sig figs for uncertainty."""
    if unc == 0 or np.isnan(unc):
        return f"{val:.6g} ± {unc:.6g}"

    exp = int(np.floor(np.log10(abs(unc))))
    unc_rounded = round(unc, -exp + (sigfigs - 1))
    decimals = max(-int(np.floor(np.log10(unc_rounded))) + (sigfigs - 1), 0)
    val_rounded = round(val, decimals)

    return f"{val_rounded:.{decimals}f} ± {unc_rounded:.{decimals}f}"

avg_dict = {
    f"px ({xy_unit})": format_val_unc(px,px_unc),
    f"py ({xy_unit})": format_val_unc(py,py_unc),
    "dx (" + z_unit+")": format_val_unc(params[0],param_unc[0]),
    "dy (" + z_unit+")": format_val_unc(params[1],param_unc[1]),
    "dt (" + z_unit+")": format_val_unc(params[2],param_unc[2]),
    "average height (" + z_unit+")": format_val_unc(params[3],param_unc[3]),
    "wx (" + xy_unit+")": format_val_unc(params[4],param_unc[4]),
    "wy (" + xy_unit+")": format_val_unc(params[5],param_unc[5])
}
if offset:
    avg_dict["Offset (" + z_unit + ")"] = format_val_unc(np.nanmean(plane_vals),np.nanstd(plane_vals))

print("\nAverage parameters of the pyramids: ")
print(tabulate.tabulate([avg_dict],headers="keys",tablefmt='tsv'))

errbar(np.linspace(0,(xycalibration*x_cut.size),x_cut.size, endpoint=False),x_cut,x_cut_unc,"Average Pyramid x cross-section", "x (" + str(xy_unit) + ")", "z (" + str(z_unit) + ")")
errbar(np.linspace(0,(xycalibration*y_cut.size),y_cut.size, endpoint=False),y_cut,y_cut_unc,"Average Pyramid y cross-section", "y (" + str(xy_unit) + ")", "z (" + str(z_unit) + ")")
errbar(np.linspace(0,(np.sqrt(2)*xycalibration*diagonal_cut.size),diagonal_cut.size, endpoint=False),diagonal_cut,diagonal_cut_unc,"Average Pyramid diagonal cross-section", "y (" + str(xy_unit) + ")", "z (" + str(z_unit) + ")")

#extra analysis
'''plt.figure()
avg_params = np.array(avg_params)
plt.scatter((np.arange(0,avg_params.shape[0]) % cols),avg_params[:,0])
plt.title("Pyramid Height vs Pyramid Position", fontsize = 24)
plt.xlabel("Pyramid Number, From Left to Right", fontsize = 22)
plt.ylabel("Pyramid Height (" + z_unit +")", fontsize = 22)'''

plt.show()