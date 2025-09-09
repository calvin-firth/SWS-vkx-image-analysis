import astropy.modeling.fitting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import scipy
from scipy.linalg import lstsq
import Load_vk4
import pandas
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler

def get_pitch(img,xycalibration, xyunit, zeropeak = False, target = 300,plot=False):
    x = np.arange(0, img.shape[1], 1)
    x_og = xycalibration * x
    # print(x)
    y = np.arange(0, img.shape[0], 1)
    y_og = xycalibration * y
    # print(y)

    x, y = np.meshgrid(x_og, y_og)

    window = skimage.filters.window('hann', img.shape)
    x_fit = x.flatten()
    y_fit = y.flatten()
    z_fit = img.flatten()
    A = np.c_[x_fit, y_fit, np.ones_like(x_fit)]
    C, _, _, _ = lstsq(A, z_fit)
    detrend = C[0] * x + C[1] * y + C[2]
    # plt.figure()
    # plt.imshow(detrend, cmap=cm.coolwarm, interpolation='nearest')
    detrend = img - detrend
    detrend = detrend - np.mean(detrend)
    # img = detrend
    # plt.figure()
    # plt.imshow(detrend,vmin = -1000, vmax=1000, cmap=cm.coolwarm, interpolation='nearest')
    data_to_ft = detrend * window
    padding_amty =  100
    padding_amtx =  100
    data_to_ft = np.pad(data_to_ft, (padding_amtx, padding_amty), 'constant')

    kx = np.fft.fftfreq(data_to_ft.shape[1], xycalibration)
    ky = np.fft.fftfreq(data_to_ft.shape[0], xycalibration)
    kx, ky = np.meshgrid(kx, ky)
    FT_2d = np.fft.fft2(data_to_ft)
    kx_q1_0 = kx[0:(int(FT_2d.shape[0] / 2))]
    ky_q1_0 = ky[0:(int(FT_2d.shape[0] / 2))]
    kx_q1 = np.zeros((FT_2d.shape[0] // 2, FT_2d.shape[1] // 2))
    ky_q1 = np.zeros((FT_2d.shape[0] // 2, FT_2d.shape[1] // 2))
    quad_1_0 = FT_2d[0:(int(FT_2d.shape[0] / 2))]
    quad_1FT = np.zeros((FT_2d.shape[0] // 2, FT_2d.shape[1] // 2), dtype=np.complex128)
    for i in range(quad_1FT.shape[0]):
        quad_1FT[i] = quad_1_0[i][0:quad_1FT.shape[1]]
        kx_q1[i] = kx_q1_0[i][0:quad_1FT.shape[1]]
        ky_q1[i] = ky_q1_0[i][0:quad_1FT.shape[1]]

    # fig2,ax2=plt.subplots()
    # levels = np.linspace(np.log10(np.abs(FT_2d)).min(),np.log10(np.partition((np.abs(FT_2d)).flatten(), -2)[-10]),50)
    # print(levels)
    # surf = ax2.contourf(kx,ky,np.log10(np.abs(FT_2d)), levels=levels,extend='both')
    # fig2.colorbar(surf, shrink=0.5, aspect=5)
    # ax2.view_init(90,0)
    # plt.title("2D Fourier Transform",fontsize=24)
    # plt.xlabel("kx (1/" + str(xy_unit) +")",fontsize=22)
    # plt.ylabel("ky (1/" + str(xy_unit) +")",fontsize=22)

    max_finder = np.array(np.abs(quad_1FT))
    max_finder[0][0] = 0
    max_finder[1][0] = 0
    max_finder[2][0] = 0
    max_finder[0][1] = 0
    max_finder[0][2] = 0

    x_m_indices = np.zeros(5, dtype=int)
    y_m_indices = np.zeros(5, dtype=int)
    if(zeropeak):
        max_finder[0] = np.where(((1/(target + 0.015) < kx_q1[0])*(1/(target-0.015) > kx_q1[0])),max_finder[0],0)
        y_finder = np.where(((1/(target + 0.05) < ky_q1[:,0])*(1/(target-0.05) > ky_q1[:,0])),np.transpose(max_finder)[0],0)
        x_m_indices[0] = np.argmax(max_finder[0])
        y_m_indices[0] = np.argmax(y_finder)

        plt.figure()
        plt.scatter(kx_q1[0], max_finder[0])
        plt.title("x")
        plt.xlabel("1/mm")

        plt.figure()
        plt.scatter(ky_q1[:, 0], y_finder)
        plt.title("y")
        plt.xlabel("1/mm")
        plt.show()
    else:
        x_m_indices[0] = np.argmax(max_finder[0])
        y_m_indices[0] = np.argmax(np.transpose(max_finder)[0])
    '''    else:
            x_m_indices[i] = np.argmax(
                max_finder[0][int((i + 1) * (x_m_indices[0]) - 2):int((i + 1) * (x_m_indices[0]) + 2)]) + (i + 1) * \
                             x_m_indices[0] - 2
            y_m_indices[i] = np.argmax(np.transpose(max_finder)[0][
                                       int((i + 1) * (y_m_indices[0]) - 2):int((i + 1) * (y_m_indices[0]) + 2)]) + (
                                         i + 1) * (y_m_indices[0]) - 2'''

    upper = int(x_m_indices[0])
    lower = int(x_m_indices[0])
    while ((max_finder[0])[lower] > ((max_finder[0])[x_m_indices[0]]) / 10.0 or (max_finder[0])[upper] > (
    (max_finder[0])[x_m_indices[0]]) / 10.0): #CHANGE BACK TO 10
        upper += 1
        lower -= 1

    uppery = int(y_m_indices[0])
    lowery = int(y_m_indices[0])
    while ((np.transpose(max_finder)[0])[lowery] > ((np.transpose(max_finder)[0])[y_m_indices[0]]) / 10.0 or
           (np.transpose(max_finder)[0])[uppery] > ((np.transpose(max_finder)[0])[y_m_indices[0]]) / 10.0): #CHANGE BACK TO 10
        uppery += 1
        lowery -= 1


    if(plot):
        plt.figure()
        plt.plot(kx_q1[0], max_finder[0],'-o')
        plt.title("x FT cut",fontsize=22)
        plt.xlabel("1/" + str(xyunit),fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure()
        plt.plot(ky_q1[:, 0], max_finder[:,0],'-o')
        plt.title("y FT cut",fontsize=22)
        plt.xlabel("1/" + str(xyunit),fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    fit_data = (max_finder[0])[lower:upper]

    fit_datay = (np.transpose(max_finder)[0])[lowery:uppery]

    fitter = astropy.modeling.fitting.LevMarLSQFitter()
    model = astropy.modeling.models.Gaussian1D((max_finder[0])[x_m_indices[0]], kx_q1[0][x_m_indices[0]],
                                               ((kx_q1[0][(upper)] - kx_q1[0][lower]) / 4.0))
    fitted_model = fitter(model, kx_q1[0][lower:upper], fit_data)
    gauss_x = np.linspace(kx_q1[0][lower], kx_q1[0][upper], 100)
    x_unc = (np.sqrt(np.diag(fitter.fit_info['param_cov']))[1])

    # print((np.transpose(max_finder)[0])[y_m_indices[0]])
    modely = astropy.modeling.models.Gaussian1D((np.transpose(max_finder)[0])[y_m_indices[0]],
                                                np.transpose(ky_q1)[0][y_m_indices[0]], ((np.transpose(ky_q1)[0][
                                                                                              uppery] -
                                                                                          np.transpose(ky_q1)[0][
                                                                                              lowery]) / 4.0))
    fitted_modely = fitter(modely, np.transpose(ky_q1)[0][lowery:uppery], fit_datay)
    gauss_y = np.linspace(np.transpose(ky_q1)[0][lowery], np.transpose(ky_q1)[0][uppery], 100)
    y_unc = (np.sqrt(np.diag(fitter.fit_info['param_cov'])[1]))

    if(plot):

        plt.figure()
        plt.scatter(kx_q1[0][lower:upper],fit_data)
        plt.plot(gauss_x,fitted_model(gauss_x))
        plt.title("x FT Gaussian fit",fontsize=22)
        plt.xlabel("1/" + str(xyunit),fontsize=18)
        plt.figtext(0.65,0.7,"$\\sigma = $" + "{:.2g}".format(fitted_model.stddev.value) + "\n$\\delta \\mu$ = " + "{:.2g}".format(x_unc),fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure()
        plt.scatter(ky_q1[lowery:uppery,0],fit_datay)
        plt.plot(gauss_y, fitted_modely(gauss_y))
        plt.title("y FT Gaussian fit",fontsize=22)
        plt.xlabel("1/" + str(xyunit),fontsize=18)
        plt.figtext(0.65, 0.7,
                    "$\\sigma = $" + "{:.2g}".format(fitted_modely.stddev.value) + "\n$\\delta \\mu$ = " + "{:.2g}".format(
                        y_unc), fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.show()
    px = 1 / (fitted_model.mean.value)
    px_unc = (x_unc) / (fitted_model.mean.value ** 2)

    py = 1 / (fitted_modely.mean.value)
    py_unc = (y_unc) / (fitted_modely.mean.value ** 2)

    print("px = " + str(px) + " +- " + str(px_unc) + " " + xyunit)
    print("py = " + str(py) + " +- " + str(py_unc) + " " + xyunit)
    return px,py,px_unc,py_unc

def open_image(path):
    if (path[-3:] == "vk6" or path[-3:] == "vk4" or path == ""):
        img, xycalibration, zcalibration = Load_vk4.open_vkx(path)
        #print(Load_vk4.get_info(path))
        xy_unit = "mm"
        z_unit = "mm"
        '''path2 = str(input("Enter the path to your image: "))
        if (path2[0] == "\""):
            path2 = path2[1:]
        if (path2[-1] == "\""):
            path2 = path2[:-1]
        img2 = pandas.read_csv(path2, skiprows=15, header=None)
        img2 = img2.to_numpy()
        plt.imshow(img2 - (1000*img))'''

    elif (path[-3:] == "csv"):
        # hdr = pandas.read_csv(path, nrows = 15,header=None)
        hdr1 = pandas.read_csv(path, skiprows=6, nrows=1, header=None)
        hdr2 = pandas.read_csv(path, skiprows=12, nrows=1, header=None)
        # print(hdr1)
        xycalibration = float(hdr1.iloc[0][1])
        # print(xycalibration)
        xy_unit = hdr1.iloc[0][2]
        # print(xy_unit)
        z_unit = hdr2.iloc[0][1]

        img = pandas.read_csv(path, skiprows=15, header=None)
        img = img.to_numpy()

    else:
        print("Unsupported filetype")
        exit()

    return img, xycalibration, xy_unit, z_unit

def partial_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Ablation Image", fontsize=24)
    plt.xlabel("x", fontsize=22)
    plt.ylabel("y", fontsize=22)
    plt.figtext(0.6, 0.5, "Click four points to define the\ntop, right, bottom, and left limits\n(in that order) of the part\n of the image to analyze", fontsize=17,
                backgroundcolor='white')
    zoom_factory(ax)
    ph = panhandler(fig, button=3)
    points = clicker(ax, ["click"], colors='k', disable_legend=True)
    plt.show()

    # User interface

    if not points.get_positions()['click'].any():
        print("You didn't place any points!! Try Again.")
        quit()

    left = points.get_positions()['click'][-1]
    bottom = points.get_positions()['click'][-2]
    right = points.get_positions()['click'][-3]
    top = points.get_positions()['click'][-4]

    '''fig2, ax2 = plt.subplots()
    ax2.imshow(img)
    ax2.scatter(top[0], top[1], s=20, marker='*', c='k')
    ax2.scatter(right[0],right[1], s=20, marker='x', c='k')
    ax2.scatter(bottom[0],bottom[1], s=20, marker='o', c='k')
    ax2.scatter(left[0],left[1], s=20, marker='.', c='k')
    plt.show()'''

    img = img[int(top[1]):int(bottom[1]), int(left[0]):int(right[0])]
    '''plt.figure()
    plt.imshow(img)
    plt.show()'''
    return img
