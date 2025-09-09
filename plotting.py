import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(x, y, title= "", xlabel = "", ylabel = ""):
    plt.figure()
    plt.scatter(x, y)
    plt.title(title, size = 24)
    plt.xlabel(xlabel, size = 22)
    plt.ylabel(ylabel, size = 22)
    plt.xticks(size=22)
    plt.yticks(size=22)

def line_plot(x, y, title = "", xlabel = "", ylabel = ""):
    plt.figure()
    plt.plot(x, y)
    plt.title(title, size = 24)
    plt.xlabel(xlabel, size = 22)
    plt.ylabel(ylabel, size = 22)
    plt.xticks(size=22)
    plt.yticks(size=22)

def errbar(x, y, yerr,title = "", xlabel = "", ylabel = ""):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr)
    plt.title(title, size = 24)
    plt.xlabel(xlabel, size = 22)
    plt.ylabel(ylabel, size = 22)
    plt.xticks(size=22)
    plt.yticks(size=22)
