#!/usr/bin/env python3

"""
bentPlumeAnalyser.py

A set of utilities to analyse wind-affected (bent) plumes

Provides functions:
-------------------
- centroid_posn
    Locate the "centre of mass" and spread of a plume.
- plume_trajectory
- gaussian_profile
- show_scaled_image    
- open_plot_expt_image
    [No documentation currently available]
- dist_along_path
    Calculates the cumulative distance along the plume axis.
- plume_angle
    Calculates the plume angle at each point along its axis.
- initial_guess_at_axis
- rotate_image
    rotates an image by a given angle.
- true_location_width    
- path_from_smoothed_theta
- pixel_to_world_posns
- world_to_pixel_posns
"""
from scipy.io.matlab import loadmat
from scipy.interpolate import interp1d, splrep, splev
from scipy.optimize import curve_fit


import numpy as np
import matplotlib.pyplot as plt
import json 


# scale_factor = 38.               # Conversion factor/[pix/cm]


def centroid_posn(x, y, n=2):
    '''
    Locate the "centre of mass" and spread of a plume, following [1]

    Parameters
    ----------
    x : array like
        the positions at which the data is sampled
    y : array like
        the array of data
    n : scalar (optional)
        Exponent to which data are raised, default is 2.

    Returns
    -------
    COM : float
        Centre of mass of the array y given by (x * y**n).sum() / (y**n).sum()

    References
    ----------
    [1] D. Contini, A.Donateo, D.Cesari, A.G.Robins (2011), "Comparison of 
    plume rise models against water tank experimental data for neutral and 
    stable cross flows", J. Wind Eng. Ind. Aerodyn., 99, pp. 539--553    
    '''
    x = np.array(x)
    y = np.array(y)

    return (x * y**n).sum() / (y**n).sum()


def plume_trajectory(image, n=2, theta=np.pi/2, scale_factor=1):
    """
    Locate the "centre of mass" and spread of a plume, following [1]

    Parameters
    ----------
    image : numpy-array like
        Plume image.
    n : scalar
        Exponent to which rows and cols are raised, default is 2.
    theta : float
        Initial plume angle.  Default is np.pi / 2 (starts vertically)
    scale_factor : float
        Scaling between pixels and real-world units.  Default is 1.

    Returns
    -------
    xbar, zbar : numpy-array like
        the centre of mass of the plume, following [1]
    x0, z0 : scalars
        Location of the vent in terms of the image offset 

    References
    ----------
    [1] D. Contini, A.Donateo, D.Cesari, A.G.Robins (2011), "Comparison of 
    plume rise models against water tank experimental data for neutral and 
    stable cross flows", J. Wind Eng. Ind. Aerodyn., 99, pp. 539--553
    """
    # Ensure that image is a numpy array
    image = np.array(image)

    # Start by running over the rows of the image by fitting a gaussian to
    # the intensity profile, until either the plume angle is less than 60
    # degrees, or the fitting routine fails.
    x, z = np.arange(image.shape[1]), np.arange(image.shape[0])
    xbar, xsig, zbar, zsig = [], [], [], []
    p0 = (.8, 50, 50)

    for pos, row in enumerate(image):
        if any(row):
            xbar.append(centroid_posn(x, row))
            zbar.append(pos)
            # try:
                # popt, pcov = curve_fit(gaussian_profile, x, row, p0)
                # xbar.append(popt[1])
                # xsig.append(popt[2])        
                # zbar.append(pos)
            if len(xbar) > 1:
                dx = np.abs(xbar[-1] - xbar[-2])
                dz = np.abs(zbar[-1] - zbar[-2])
                theta = np.arctan(np.divide(dz, dx))
            if theta < np.pi / 3: # 60 degrees...
                break
            # except RuntimeError:
            #     break
            
    # Split the plume image into what remains beyond the break point of the
    # previous loop.
    x0    = int(np.ceil(xbar[-1])) # Move to the next point to the right
    image = image[:, x0:]

    # Iterate over columns of the image, but only do so if the plume is
    # present at that location.
    p0 = (.8, image.shape[0] / 2, 50)
    for pos, col in enumerate(image.T):
        if any(col):
            # Calculate weighted mean and std dev over the col
            xbar.append(pos + x0)
            zbar.append(centroid_posn(z, col))
            # zsig.append(np.sqrt(((z - zbar[-1])**2 * col).sum() / col.sum()))
            # try:
            #     popt, pcov = curve_fit(gaussian_profile, z, col, p0)
            #     xbar.append(pos + x0)
            #     zbar.append(popt[1])
            # except RuntimeError:
            #     break
    xbar = np.array(xbar)
    zbar = np.array(zbar)
    x0 = (xbar[0], zbar[0])
    xbar = (xbar - x0[0]) / scale_factor
    zbar = (zbar - x0[1]) / scale_factor
    return xbar, zbar, x0
    

def gaussian_profile(x, offset, amplitude, loc, width):
    """
    Returns an offset gaussian profile defined by its amplitude, 
    """
    return offset + amplitude * np.exp(-.5 * ((x - loc) / width)**2)


def show_scaled_image(image, scale_factor=1., vent_loc=None,
                      ax=None, cmap=plt.cm.gray):
    """
    
    """
    Ox, Oz = 0, 0
    if vent_loc is not None:
        Ox, Oz = vent_loc
    
    extentInPix = np.array([0, image.shape[1], 0, image.shape[0]])
    extent = np.array([(extentInPix[:2] - Ox) / scale_factor,
                       (extentInPix[2:] - Oz)[::-1] / scale_factor]).flatten()
    
    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(image, extent=extent, cmap=cmap)

    return extent, im, ax


def open_plot_expt_image(path, axes, scale_factor=1., exptNo=1,
                         ind=None, showPlot=True):
    """
    
    """
    fname = path + 'gsplume.mat'
    
    data = np.flipud(loadmat(fname)['gsplume'])
    xexp = loadmat(path + 'xcenter.mat')['xcenter'][0]
    zexp = loadmat(path + 'zcenter.mat')['zcenter'][0]
    
    # Invert the data image if the background is brighter than background
    # (supposing that the background dominates)
    if data.mean() > .5:
        data = 1. - data

    # Coordinates of the origin as the first points in (xexp, zexp)
    Ox, Oz = (xexp[0], zexp[0])
    # Now calculate the world extent of the data, using a conversion factor 
    # of 38 pixels to 1 cm.  There's probably some neater and more pythonic
    # way of calculating the world extent list but at least it works as is.
    extentInPix = [0, data.shape[1], 0, data.shape[0]]
    extent = np.array([(extentInPix[:2] - Ox) / scale_factor,
                       (Oz - extentInPix[2:]) / scale_factor]).flatten().tolist()
    xexp = (xexp - Ox) / scale_factor
    zexp = (Oz - zexp) / scale_factor

    xbar, zbar, x0 = plume_trajectory(data, 2, np.pi/2, scale_factor)

    if ind is not None and len(axes) > 1:
        ax = axes[ind]
    else:
        ax = plt.gca() 

    im = ax.imshow(data, extent=extent, cmap=plt.cm.magma, zorder=0)
    ax.invert_yaxis()
    ax.plot(xexp, zexp, 'w-',  lw=2, label='GCTA centroid', zorder=3)
    ax.plot(xbar, zbar, 'w--', lw=2, label='my centroid', zorder=3)
    ax.set_title('Expt %2d' % exptNo, fontsize=10)

    if showPlot:
        plt.show()

    return ax, im, data, xexp, zexp, extent, (Ox, Oz)


def dist_along_path(x, y):
    """
    Returns a vector of the cumulative euclidian norms for input (x,y), i.e.
    the distance travelled along the path defined by the loci (x, y)

    """
    # Differences between consectutive points
    dx, dy = np.diff(x), np.diff(y)
    # The first point - assumes that x and y are measured with respect to
    # a well-defined origin
    s0 = np.sqrt(x[0]**2 + y[0]**2)
    # Add the first point to the cumulative sum of the differences
    return np.append(0, np.sqrt(dx**2 + dy**2).cumsum()) + s0


def plume_angle(x, y, errors=None):
    """
    Calculates the angle of a plume from the axis coordinates, (x,y), as
    $\theta = \atan(dy / dx)$.  Note that $\theta$ is measured relative to the 
    horizontal.

    parameters
    ----------
    x, y : array_like
        coordinates of the plume axis
    errors : tuple or array_like (optional)
        measurement errors on x and y.  If errors contains more than two 
        elements, an error will be raised.

    returns
    -------
    theta : array_like
        angle of the plume axis relative to the horizontal.  Positive angles 
        are measured in the anti-clockwise direction.
    sig_theta : array_like
        error associated with angle calculations

    see also
    --------
    numpy.arctan2
    """
    try:
        # Basic functionality: calculate only angle
        dy = np.gradient(y, edge_order=2)
        dx = np.gradient(x, edge_order=2)
        theta = np.arctan2(dy, dx)
    except ValueError:
        theta = None

    if errors is not None:
        # Advanced functionality: calculate measurement error
        denom     = dx**2 + dy**2
        dfdx      = - dy / denom
        dfdy      =   dx / denom
        sig_theta = np.sqrt((dfdx * errors[0])**2 + (dfdy * errors[1])**2)
        return theta, sig_theta
    else:    
        return theta


def initial_guess_at_axis(N=50):
    """
    Get a maximum of N points from the current image
    """
    # If no input for p is given (i.e. p is None), p is an empty list, or
    # p is an empty array, or any entry in p is None, then initialise p.
    # or (type(p) is list and p == []) or (type(p) is np.ndarray
    # and p.size == 0) or if any(p == None):
    print('Click on the points that will form the initial guess')
    return np.array(plt.ginput(N)) # Max of N pts



def rotate_image(img, angle, pivot):
    '''
    Rotates an image by padding the edges by an amount equal to the 
    location of the pivot point.

    Parameters
    ----------
    img : 2d array (image)
        The 2D (greyscale) image to be rotated
    angle : scalar
        Angle (in degrees) through which the image will be rotated
    pivot : array_like
        Two-element point around which the image will be rotated

    Returns
    -------
    img_rot : 2D array (image)
        The rotated image
    '''
    from skimage.transform import rotate

    pad_x = [img.shape[1] - pivot[0], pivot[0]]
    pad_y = [img.shape[0] - pivot[1], pivot[1]]
    pad_w = np.array([pad_y, pad_x]).astype(int)
    row_trim, col_trim = np.array(img.shape) // 2
    # Add left and right padding to image so as to centre the pivot location.
    # Padded image will be twice the size of the initial image array.
    img_pad = np.pad(img, pad_w, 'edge')
    img_rot = rotate(img_pad, angle, resize=False, mode='edge')

    return img_rot[row_trim:2*img.shape[0]-row_trim,
                   col_trim:2*img.shape[1]-col_trim]


def true_location_width(data, mask=None, p=None, scale_factor=1,
                        errors=None, plotting=False):
    '''
    Returns the "true" location of the plume centroid for 

    Parameters
    ----------
    data : image (NDarray)
        The plume image to be analysed.
    p : array_like (optional)
        A guess (initial or updated) as to the location of the plume axis.
        "p" should be given in pixel values
    mask : NDarray (optional)
        A mask to be applied to the image

    Returns
    -------
    true_locn : 1D array like
        The "true" location of the plume centroid
    plume_width : 1D array like
        The width at location trueLocn[i] and angle theta[i]

    See also
    --------
    plume_angle
    '''
    if mask is None:
        mask = np.ones_like(data).astype(float)

    if p is None:
        p = initial_guess_at_axis()

    s = dist_along_path(*p.T)

    
    # Iterate through the selected points, rotating the image through 
    # the estimated angle and calculating the true location of the midpoint
    # at each point.
    h_window_offset = 250
    v_window_offset =   5
    true_locn, true_loc_err, plume_width, plume_width_err = [], [], [], []
    r2 = 0.

    if plotting:
        # fig1, axes1 = plt.subplots(1, 2)
        fig1 = plt.figure()
        ax00 = fig1.add_axes([.05,       .35, .4,        .6])
        ax01 = fig1.add_axes([.61724047, .35, .26551907, .6])
        ax10 = fig1.add_axes([.05,       .05, .4,        .2])
        ax11 = fig1.add_axes([.61724047, .05, .26551907, .2])

        ax00.imshow(data) #[::-1])
        ax00.plot(p[:,0], p[:,1], 'r+', lw=2)
        
        ax10.set_xlim((s.min(), s.max()))
        ax10.axhline(0., c='k', ls='--', lw=.5)
        ax10.set_title('$r^2$: %.4f' % r2)

    # Loop through the locations in p, rotating the image as required and
    # obtaining the maximum intensity and the half width of the plume
    # "section" at each location from a Gaussian fit.
    theta, sig_theta = plume_angle(*p.T, errors=[1 / scale_factor]*2)

    # Lists for the var iance of b and d (from covariance matrix)
    var_b, var_d, d = [], [], []

    for p_, th in zip(p, theta):
        im_r = rotate_image(data, 90-np.rad2deg(th), p_)  # data[::-1]
        ma_r = rotate_image(mask, 90-np.rad2deg(th), p_)  
        M, N = im_r.shape
        M_2  = M // 2

        # Define the plume section as the 2*v_window_offset+1 (~10) rows 
        # centred about the current location, and return the row-wise mean.
        rma  = ma_r[M_2-v_window_offset:M_2+v_window_offset+1].mean(axis=0)
        row  = im_r[M_2-v_window_offset:M_2+v_window_offset+1].mean(axis=0)
        #row -= row.mean() * rma
        row *= rma

        # Fit a Gaussian to the data and use this to define plume parameters
        p0   = (0., row.max(), N/2, 50.)
        X = np.arange(len(row))
        popt, pcov = curve_fit(gaussian_profile, X, row, p0)
        plume_width.append(popt[-1])
        plume_width_err.append(np.sqrt(np.diag(pcov[-1])))
        var_b.append(pcov[-1][-1])
        # Distance from centre of plume to the centre of the row, in
        # rotated coordinates
        d.append(popt[2] - (N//2)) # COM - (N/2) #
        var_d.append(d[-1]**2 * pcov[2][2])
        # Add the distance (i.e. residual) squared to the R2 value
        r2 += d[-1]**2
        # Do some geometry to find the "true location" of the plume centre
        # Should the distance from COM be added or subtracted from estimated
        # plume axis locus?
        true_locn.append(np.float64(p_) -   # Should this be a + or -?
                        np.multiply(d[-1], [np.cos(th), np.sin(th)]))

        # ----------------------------------------------------------------- #
        # The remainder of this function is dedicated to plotting the
        # solutions.  Should probably incorporate a boolean handle that turns
        # this functionality on or off.
        # ----------------------------------------------------------------- #
        if plotting:
            # Plot the solution for visualisation purposes.
            # Update "true locations" on the base image
            ax00.plot(true_locn[-1][0], true_locn[-1][1], 'c+', ms=8, lw=3)
            
            # Rotate the image to show where current section is being calculated
            ax01.imshow(im_r)
            ax01.axvline(N/2)
            ax01.axvline(N/2 - h_window_offset, ls='--')
            ax01.axvline(N/2 + h_window_offset, ls='--')
            ax01.axhline(M/2)
            
            # Plot the residual as a function of distance along plume axis
            #ax10.plot(s, d[-1], '+', c='C0')
            #ax10.set_title('$r^2$: %.4f' % r2)
            
            # Plot the section for the current rotation
            ax11.clear()
            ax11.plot(X, row, '-', c='C0', label='data')
            ax11.plot(X, gaussian_profile(X, *popt), 'r-', label='gaussian')
            ax11.axvline(popt[2], c='r', ls='--', lw=2, zorder=1)
            ax11.axvline(N/2, c='C0', ls='-', zorder=0)
            ax11.axvline(N/2 - h_window_offset, c='C0', ls='--', zorder=0)
            ax11.axvline(N/2 + h_window_offset, c='C0', ls='--', zorder=0)
            plt.pause(.5)  # Pause for a crude animation
        #plt.close()

    # Convert lists to numpy arrays for ease of manipulation
    var_b = np.array(var_b)
    var_d = np.array(var_d)
    d = np.array(d)
    if errors is not None:
        sig_true_locn = np.sqrt(2 * errors[0]**2 + var_d + d**2 * sig_theta)
        return (np.array(true_locn), np.abs(np.array(plume_width)),
                sig_true_locn, np.sqrt(var_b))
    else:
        return np.array(true_locn), np.abs(np.array(plume_width))


def path_from_smoothed_theta(s, theta, snew, smoothing=0.): 
    '''
    Returns a smoothed version of the axis locus
    '''
    tck = splrep(s, theta, s=smoothing) 
    thetaNew = splev(snew, tck) 
    xnew, znew = [0], [0] 
    for ds, th in zip(np.diff(snew), thetaNew): 
        dx = ds * np.cos(th) 
        dz = ds * np.sin(th) 
        xnew.append(xnew[-1] + dx) 
        znew.append(znew[-1] + dz) 
    return np.array(xnew), np.array(znew), thetaNew


def pixel_to_world_posns(pixel_posns, offset, scale_factor=1):
    return (pixel_posns - offset) * [1, -1] / scale_factor


def world_to_pixel_posns(world_posns, offset, scale_factor=1):
    return world_posns * [1, -1] * scale_factor + offset
    

if __name__ == '__main__':
    plt.close('all')    # Clear pre-existing plots

    # Set a path variable according to which experiment is required
    exptNo = 3
    path   = './data/ExpPlumes_for_Dai/exp%02d/' % exptNo
    #fname  = path + 'gsplume.mat'
    #path = '/home/ovsg/Documents/Thermographie/done/tiff/2019/mu sigma/'

    # Initial guesses.  Either load from file or make a fresh guess
    with open(path + 'exp%02d_initGuess.json' % exptNo) as f:
    #with open(path + '20190314_CSS_initGuess.json') as f:
        jsondata = json.load(f)
    p = np.array(jsondata['data'])

    scale_factor = 77.44015387936139
    
    fig, ax = plt.subplots()    
    try:
        axes = np.array([ax])
        ax, im, data, xexp, zexp, extent, (Ox, Oz) = open_plot_expt_image(path,
                                                                       axes),
                                                                       #exptNo)
        leg  = ax.legend()
        leg.get_frame().set_facecolor('gray')                                   
    except TypeError:
        print('file %s doesn\'t seem to exist...skipping!' % fname)

    #p       = initial_guess_at_axis()  # Comment out if loading from file
    pPixels = p.copy() * scale_factor
    pPixels[:,0] += Ox
    pPixels[:,1] -= Oz
    pPixels[:,1] *= -1

    thexp, sig_thexp = plume_angle(*p.T, errors=[1/scale_factor]*2)
    _, bexp, sig_p, sig_bexp = true_location_width(pPixels, data,
                                                 errors=[1/scale_factor],
                                                 plotting=True)

    # Form the state variables that will be compared to model solution. 
    #p[:,0] = (p[:,0] - Ox) / scale_factor
    #p[:,1] = (Oz - p[:,1]) / scale_factor
    bexp     /= scale_factor
    sig_bexp /= scale_factor
    sig_p    /= scale_factor
    sexp = dist_along_path(p[:,0], p[:,1])
    Vexp = np.array([bexp, thexp]).T
    sigV = np.array([sig_bexp, sig_thexp]).T

    # fname  = './data/ExpPlumes_for_Dai/GCTA_plumeData.xlsx'
    # writer = pandas.ExcelWriter(fname, engine='xlsxwriter') 
