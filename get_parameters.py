import numpy as np
import matplotlib.pyplot as plt
from plotting import line_plot

def get_params(pyr, x, y, plot = False, offset = False,plane = [],num_pyr = 0):
    top_fraction = 0.333
    x_cut = pyr[pyr.shape[0] // 2]
    y_cut = np.transpose(pyr)[pyr.shape[1] // 2]
    diagonal_cut = np.diagonal(pyr)

    x_min = (np.argmin(x_cut), np.min(x_cut))
    x_top = np.where(x_cut > (0.95 * np.ptp(x_cut)) + x_min[1])
    x_top_span = x_cut[x_top].size
    x_peakind = x_top[0][int(0.5*(1 - top_fraction)*x_top_span):int(0.5*(1+top_fraction)*x_top_span)]
    x_peak = x_cut[x_peakind]
    x_p = x[x_peakind]

    x_h = np.median(x_peak) - x_min[1]

    x_top = np.where(x_cut > (0.95 * x_h) + x_min[1])

    y_min = (np.argmin(y_cut), np.min(y_cut))
    y_top = np.where(y_cut > (0.95 * np.ptp(y_cut)) + y_min[1])
    y_top_span = y_cut[y_top].size
    y_peakind = y_top[0][int(0.5*(1 - top_fraction) * y_top_span):int(0.5 * (1 + top_fraction) * y_top_span)]
    y_peak = y_cut[y_peakind]
    y_p = y[y_peakind]
    y_h = np.median(y_peak) - y_min[1]

    y_top = np.where(y_cut > (0.95 * y_h) + y_min[1])

    d_min = (np.argmin(diagonal_cut), np.min(diagonal_cut))
    d_top = np.where(diagonal_cut > (0.95 * np.ptp(diagonal_cut)) + d_min[1])
    d_top_span = diagonal_cut[d_top].size
    d_peakind = d_top[0][int(0.5 * (1 - top_fraction) * d_top_span):int(0.5 * (1 + top_fraction) * d_top_span)]
    d_peak = diagonal_cut[d_peakind]
    d_p = np.sqrt(2)*y[d_peakind]

    d_h = np.median(d_peak) - d_min[1]

    if(offset):
        plane[num_pyr] = plane[num_pyr] - (np.median(x_peak) + np.median(y_peak) + np.median(d_peak))/3

    #plotting
    if(plot):
        line_plot(x[:x_cut.size], x_cut,"x cross-section", "x", "z")
        plt.scatter(x[x_min[0]], x_min[1])
        plt.scatter(x[x_top], x_cut[x_top])
        plt.scatter(x_p, x_peak)
        plt.axhline(y=np.median(x_peak))

        line_plot(np.sqrt(2)*x[:diagonal_cut.size], diagonal_cut, "diagonal cross-section", ylabel="z")
        plt.scatter(np.sqrt(2)*x[d_min[0]], d_min[1])
        plt.scatter(np.sqrt(2)*x[d_top], diagonal_cut[d_top])
        plt.scatter(d_p, d_peak)
        plt.axhline(y=np.median(d_peak))

        line_plot(y[:y_cut.size], y_cut, "y cross-section", "y", "z")
        plt.scatter(y[y_min[0]], y_min[1])
        plt.scatter(y[y_top], y_cut[y_top])
        plt.scatter(y_p, y_peak)
        plt.axhline(y=np.median(y_peak))

        plt.show()

    if(x_top_span <= 1/(2*top_fraction) or y_top_span <= 1/(2*top_fraction) or d_top_span <= 1/(2*top_fraction) or d_peak.size == 0 or x_peak.size==0 or y_peak.size==0):
        print("Warning: A pyramid with an unexpected \"spike\" was detected. Measurements of this pyramid are not included in the data.")
        return(np.full(6,np.nan))
    else:
        return([x_h, y_h, d_h, (((x_h+y_h)/2+d_h)/2), (x[x_top][-1]-x[x_top][0]),(y[y_top][-1]-y[y_top][0])])