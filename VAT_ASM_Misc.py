import numpy as np
import os
from numba import njit, prange


def airfoil_data_parser():
    """ parse .txt files in the corresponding folders. cl and cd arrays are split in case different alpha values are used.
    the routine reads the Re number from the filenames and expects a "_ReXXXXXX_" somewhere.
    Note that cl and cd have to use the same Re numbers as the Re array is shared between them

    Args:
        no arguments. all .txt files are read from the folders "airfoil_data/cl" and "airfoil_data/cd". Must contain "_ReXXXXXX_" in the name!

    Returns:
        Re_sorted: sorted array of all available Re numbers
        cl_sorted, cd_sorted: list with array entries. Each Re from the Re_sorted corresponds to one 2D array containing alpha and cl/cd
    """

    # search for "Re" keyword in found files
    def parse(path):
        Re_list = []
        data_list = []
        all_files = os.listdir(path)
        for file in all_files:
            Re = file.split("_")
            for entry in file.split("_"):
                if entry.find("Re") != -1:
                    Re = entry.split("Re")[1]

            Re_list.append(int(Re))
            data_list.append(np.loadtxt(path + "/" + file, dtype=np.float64))

        # sort lists by Re array
        data_list = np.array(data_list, dtype="object")
        inds = np.array(Re_list).argsort()
        sorted_list = data_list[inds]
        Re_sorted = np.array(Re_list)[inds]
        return Re_sorted, sorted_list

    # approximate slope between -5 to +5 degrees AoA
    def calc_slope(cl_sorted):
        cl_slope = []
        for i in range(0,len(cl_sorted)):
            cl_data = cl_sorted[i]
            dy = linear_interp(5, cl_data[:, 0], cl_data[:, 1]) - linear_interp(-5, cl_data[:, 0], cl_data[:, 1])
            dx = 10
            cl_slope.append(dy / dx)
        return np.array(cl_slope)

    # parse data for lift and drag data seperatly
    paths = ["airfoil_data/cl", "airfoil_data/cd"]
    Re_sorted, cl_sorted = parse(paths[0])
    Re_sorted, cd_sorted = parse(paths[1])
    print("Files for Re = " + str(Re_sorted) + " found")

    # find cl slope at zero alpha for each file. This is needed by most dynamic stall models
    cl_slope = calc_slope(cl_sorted)

    return Re_sorted, cl_sorted, cd_sorted, cl_slope


def airfoil_data(alpha, Re, airfoil, return_cl=False, return_cd=False, return_slope = False):
    """ double interpolation for the lift and drag coefficients from the given alpha/coefficients files. First the two closest available Re numbers in the files are detected.
    For both files the aerodynamic coefficients are interpolated by the given the current flow angle. Finally, the current Re is used to interpolate between these two files.

    Args:
        alpha: current angle to evaluate aerodynamic coefficents for
        Re: current Reynolds-number
        airfoil: contains the pre-loaded lists of all Re numbers and the coefficients. See return from "airfoil_data_parser"

    Returns:
        cl: lift coefficient
        cd: drag coefficient
    """

    Re_all, cl_all, cd_all, slope_all = airfoil

    # stay within limits of available data. No extrapolation
    if Re <= Re_all[0]:
        Re = Re_all[0] + 1
    if Re >= Re_all[-1]:
        Re = Re_all[-1] - 1

    idx = searchsorted(Re_all, Re)

    # linear interpolation between angles
    def interpolation(xval, i, x, y):
        return y[i - 1] * (x[i] - xval) / (x[i] - x[i - 1]) + y[i] * (xval - x[i - 1]) / (x[i] - x[i - 1])

    def get_coeff(idx, alpha, coeff_all):
        idx_l = searchsorted(coeff_all[idx - 1][1:, 0], alpha)
        coeff_l = interpolation(alpha, idx_l, coeff_all[idx - 1][1:, 0], coeff_all[idx - 1][1:, 1])

        idx_r = searchsorted(coeff_all[idx][1:, 0], alpha)
        coeff_r = interpolation(alpha, idx_r, coeff_all[idx][1:, 0], coeff_all[idx][1:, 1])

        coeff = coeff_l * (Re_all[idx] - Re) / (Re_all[idx] - Re_all[idx - 1]) + \
            coeff_r * (Re - Re_all[idx - 1]) / (Re_all[idx] - Re_all[idx - 1])
        return coeff

    def get_slope(idx, slope_all):
        return interpolation(Re, idx, Re_all, slope_all)
    
    if return_slope == True:
        return get_slope(idx, slope_all)
        
    if return_cl == True and return_cd == True:
        cl = get_coeff(idx, alpha, cl_all)
        cd = get_coeff(idx, alpha, cd_all)
        return cl, cd
    elif return_cl == True and return_cd == False:
        cl = get_coeff(idx, alpha, cl_all)
        return cl
    elif return_cl == False and return_cd == True:
        cd = get_coeff(idx, alpha, cd_all)
        return cd
    else:
        raise ValueError("neither lift or drag values have been set to be returned")


@njit(nopython=True)
def searchsorted(array, element):
    """ super fast numba binary search. 6x faster than numpy.searchsorted.
        Find the indices into a sorted array as such that, if the corresponding
        element in "element" were inserted before the indices, the order of "array" would be preserved.
    """
    mid = 0
    start = 0
    end = len(array)
    step = 0

    while (start <= end):
        step = step + 1
        mid = (start + end) // 2

        if element == array[mid]:
            return mid

        if element < array[mid]:
            end = mid - 1
        else:
            start = mid + 1
    return start


@njit(nopython=True)
def linear_interp(xval, x, y):
    """ Fast O(n) linear interpolation using numba for single scalar values
    """
    for i in range(1, len(x) - 1):
        if x[i] > xval:
            break
        i = i + 1
    return y[i - 1] * (x[i] - xval) / (x[i] - x[i - 1]) + y[i] * (xval - x[i - 1]) / (x[i] - x[i - 1])


@njit(nopython=True)
def bilinear_interp(xval, yval, x, y, u):
    """ bilinear interpolation for the flow velocity components along the struts. The four corners of the mesh around the
        given point are used to 2D interpolate the velocity components.
    Args:
        xval, yval: x,y - coordinates to evaluate. Can contain more than one coordinate-set
        x, y: one dimensional mesh coordinates along each axis
        u: 2D array to evaluate the data from. In this case for example the u-velocity component

    Returns:
        results: interpolated velocity component
    """

    results = np.copy(xval)

    for n in range(0, len(xval)):
        for i in range(1, len(y) - 1):
            if y[i] > yval[n]:
                break
            i = i + 1

        for j in range(1, len(x) - 1):
            if x[j] > xval[n]:
                break
            j = j + 1

        Q11 = u[i - 1, j - 1]
        Q21 = u[i - 1, j]
        Q12 = u[i, j - 1]
        Q22 = u[i, j]

        y2 = y[i]
        y1 = y[i - 1]
        x2 = x[j]
        x1 = x[j - 1]

        a = (x2 - x1) * (y2 - y1)

        results[n] = (Q11 / a * (x2 - xval[n]) * (y2 - yval[n]) +
                      Q21 / a * (xval[n] - x1) * (y2 - yval[n]) +
                      Q12 / a * (x2 - xval[n]) * (yval[n] - y1) +
                      Q22 / a * (xval[n] - x1) * (yval[n] - y1))

    return results


def circular_mask(nx, ny, radius, center=None):
    """ create mask for tower velocity
    """
    if center is None:  # use the middle of the array
        center = (int(nx / 2), int(ny / 2))

    Y, X = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius

    return mask


def colorbar(mappable):
    """ magic colorbar trick for square plots
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
