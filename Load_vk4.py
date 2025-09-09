import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import binascii
import ipywidgets as widgets
from scipy.optimize import curve_fit
import sys
from zipfile import ZipFile
from mpl_toolkits.mplot3d import Axes3D
from os import remove
import vk4extract


class VK4_Lib:
    """
    This class is written by H. Sakurai
    """

    def getscale(self):
        with open(self.fname, 'rb') as f:
            header = f.read(264)
        XY = int(binascii.hexlify(header[252:256][::-1]), 16) / 1e9
        Z = int(binascii.hexlify(header[260:264][::-1]), 16) / 1e9
        return XY, Z

    def heightviewer(self):
        with open(self.fname, 'rb') as f:
            header = f.read(40)
        with open(self.fname, 'rb') as f:
            offset = int(binascii.hexlify(header[36:40][::-1]), 16)
            f.seek(offset)
            pre_data = f.read(28)
            width = int(binascii.hexlify(pre_data[0:4][::-1]), 16)
            height = int(binascii.hexlify(pre_data[4:8][::-1]), 16)
            databytes = int(int(binascii.hexlify(pre_data[8:12][::-1]), 16) / 8)
            totlength = int(binascii.hexlify(pre_data[16:20][::-1]), 16)
            LZWtable = f.read(256 * 3)
            data = f.read(width * height * databytes)
        data_array = np.empty(width * height)
        for i in range(width * height):
            data_array[i] = int(binascii.hexlify(data[i * databytes:(i + 1) * databytes][::-1]), 16)
        image = np.reshape(data_array, (height, width))
        return image


class Load_vk4Data(VK4_Lib):
    def __init__(self, fname):
        """
        Added by R. Takaku (U Tokyo)
        """
        self.fname = fname
        self.xy_cal = self.getscale()[0]
        self.Z_cal = self.getscale()[1]
        #print('xy resolution [mm] = ', self.xy_cal)
        #print(self.Z_cal)
        self.m = self.heightviewer()
        self.m = (self.m) * self.Z_cal

        self.x_com = len(self.m)
        self.y_com = len(self.m[0])

        self.x = np.arange(0, self.x_com * self.xy_cal, self.xy_cal)
        self.y = np.arange(0, self.y_com * self.xy_cal, self.xy_cal)

        self.X, self.Y = np.meshgrid(self.y, self.x)

def open_vkx(filestr):
    f = open("temp.txt", "a+")
    f.seek(0)
    fname = filestr or f.read()
    f.seek(0)
    if len(f.read()) == 0:
        f.write(fname)
        f.close()
    else:
        f.close()

    if fname[-4:] == '.vk4':
        shape_lib = Load_vk4Data(fname)
    elif fname[-4:] == '.vk6':

        with ZipFile(fname, 'r') as edgar:
            fname1 = edgar.extract('Vk4File')
        shape_lib = Load_vk4Data(fname1)
        remove('Vk4File')

    else:
        print("Unsupported filetype")
        sys.exit()

    data = shape_lib.m
    x_mesh = shape_lib.X
    y_mesh = shape_lib.Y

    x_1D = shape_lib.x
    y_1D = shape_lib.y

    '''fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = ax.pcolormesh(x_mesh,y_mesh,data,cmap=plt.cm.jet)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    cb = fig.colorbar(cmap,ax = ax,shrink=1)
    cb.set_label('Depth [mm]', size = 12)

    fig.tight_layout()
    plt.show()'''
    #print(shape_lib.xy_cal)
    #print(shape_lib.Z_cal)
    return data, shape_lib.xy_cal, shape_lib.Z_cal

def get_info(filestr):
    with open(filestr, 'rb') as in_file:
        offsets = vk4extract.extract_offsets(in_file)
        info = vk4extract.extract_measurement_conditions(offsets,in_file)
        return info