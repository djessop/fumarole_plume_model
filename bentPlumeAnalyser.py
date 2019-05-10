#!/usr/bin/env python3

"""
bentPlumeAnalyser.py

A set of utilities to analyse wind affected (bent) plumes

Provided functions:
-------------------
- plumeTrajectory
    Locate the "centre of mass" and spread of a plume.
- rotatedPlumeSection
    Takes a part of the experimental image, and rotates it by a given angle.
- openPlotExptImage
    [No documentation currently available]
- distAlongPath
    Calculates the cumulative distance along the plume axis.
- plumeAngle
    Calculates the plume angle at each point along its axis.
"""
from scipy.io.matlab import loadmat
from scipy.interpolate import interp1d, splrep, splev
from scipy.optimize import curve_fit


import numpy as np
import matplotlib.pyplot as plt
import json 


scaleFactor = 38.               # Conversion factor/[pix/cm]


def centroidPosn(x, y, n=2):
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


def plumeTrajectory(image, n=2):
    """
    Locate the "centre of mass" and spread of a plume, following [1]

    Parameters
    ----------
    image : numpy-array like
        Plume image.
    n : scalar
        Exponent to which rows and cols are raised, default is 2.

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
    theta = np.pi / 2           # Plume starts vertically
    for pos, row in enumerate(image):
        if any(row):
            xbar.append(centroidPosn(x, row))
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
            zbar.append(centroidPosn(z, col))
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
    xbar = (xbar - x0[0]) / scaleFactor
    zbar = (zbar - x0[1]) / scaleFactor
    return xbar, zbar, x0
    

def gaussian_profile(x, *p):
    """
    Returns a gaussian where the amplitude, centre and width are defined by
    the tuple p
    """
    return p[0] * np.exp(-(x - p[1])**2 / (2. * p[2]**2))


def openPlotExptImage(path, axes, exptNo=1, ind=None, showPlot=True):
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
    extent = np.array([(extentInPix[:2] - Ox)/scaleFactor,
                       (Oz - extentInPix[2:])/scaleFactor]).flatten().tolist()
    xexp = (xexp - Ox) / scaleFactor
    zexp = (Oz - zexp) / scaleFactor

    xbar, zbar, x0 = plumeTrajectory(data)

    if len(axes) > 1 and ind is not None:
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


def distAlongPath(x, y):
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


def plumeAngle(x, y, errors=None):
    """
    Calculates the angle of a plume from the axis coordinates, (x,y), as
    $\theta = \atan(dy / dx)$.  Note that $\theta$ is measured relative to the 
    horizontal.

    parameters
    ----------
    x, y : array-like
        coordinates of the plume axis
    errors : tuple or array-like (optional)
        measurement errors on x and y.  If errors contains more than two 
        elements, an error will be raised.

    returns
    -------
    theta : array-like
        angle of the plume axis relative to the horizontal.  Positive angles 
        are measured in the anti-clockwise direction.
    sig_theta : array-like

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
        denom = dx**2 + dy**2
        dfdx = - dy / denom
        dfdy =   dx / denom
        sig_theta = np.sqrt((dfdx * errors[0])**2 + (dfdy * errors[1])**2)
        return theta, sig_theta
    else:    
        return theta


def initialGuessAtAxis(N=50, p=None):
    """
    Get a maximum of N points from the current image
    """
    # If no input for p is given (i.e. p is None), p is an empty list, or p is an empty array,
    # or any entry in p is None, then initialise p
    if (p is None): #or (type(p) is list and p == []) or (type(p) is np.ndarray and p.size == 0) or if any(p == None):
        print('Click on the points that will form the initial guess')
        p = np.array(plt.ginput(N))        
        return p #np.append([0, 0], p).reshape(len(p)+1, 2)
    else:
        return p


def rotateImage(img, angle, pivot):
    '''
    Rotates an image by padding the edges by an amount equal to the 
    location of the pivot point.

    Parameters
    ----------
    img : 2D array (image)
        The 2D (greyscale) image to be rotated
    angle : scalar
        Angle (in degrees) through which the image will be rotated
    pivot : array like
        Two-element point around which the image will be rotated

    Returns
    -------
    imR : 2D array (image)
        The rotated image
    '''
    from skimage.transform import rotate

    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'edge')
    imgR = rotate(imgP, angle, resize=False, mode='edge') #, cval=imgP.min())

    return imgR


def trueLocationWidth(p, data, errors=None, plotting=False):
    '''
    Returns the "true" location of the plume centroid for 

    Parameters
    ----------
    p : 1D array like
        A guess (initial or updated) as to the location of the plume axis.
        "p" should be given in pixel values
    data : image (2D-array)
        The plume image to be analysed.

    Returns
    -------
    trueLocn : 1D array like
        The "true" location of the plume centroid
    plumeWidth : 1D array like
        The width at location trueLocn[i] and angle theta[i]

    See also
    --------
    plumeAngle
    '''

#print(np.stack((p[:,0], p[:,1], theta)).T)
    if p is not None:
        p = initialGuessAtAxis(p=p)
    else:
        p = initialGuessAtAxis()

    _s = distAlongPath(p[:,0], p[:,1])

    
    # Iterate through the selected points, rotating the image through 
    # the estimated angle and calculating the true location of the midpoint
    # at each point.
    hWindowOffset = 250
    vWindowOffset =   5
    trueLocn, plumeWidth = [], []
    r2 = 0.

    if plotting:
        # fig1, axes1 = plt.subplots(1, 2)
        fig1 = plt.figure()
        ax00 = fig1.add_axes([.05,       .35, .4,        .6])
        ax01 = fig1.add_axes([.61724047, .35, .26551907, .6])
        ax10 = fig1.add_axes([.05,       .05, .4,        .2])
        ax11 = fig1.add_axes([.61724047, .05, .26551907, .2])

        ax00.imshow(data[::-1])
        ax00.plot(p[:,0], p[:,1], 'r+', lw=2)
        
        ax10.set_xlim((_s.min(), _s.max()))
        ax10.axhline(0., c='k', ls='--', lw=.5)
        ax10.set_title('$r^2$: %.4f' % r2)

    # Loop through the locations in p, rotating the image as required and
    # obtaining the maximum intensity and the half width of the plume
    # "section" at each location from a Gaussian fit.
    theta, sig_theta = plumeAngle(p[:,0], p[:,1], errors=[1/scaleFactor]*2)
    # Get angles in degrees and in the correct quadrant
    theta = 90. + theta * 180 / np.pi
    # Lists for the variance of b and d (from covariance matrix)
    var_b, var_d, d = [], [], []

    for q, th, s in zip(p, theta, _s):
        q    = np.uint16(q)
        imR  = rotateImage(data[::-1], th, q)
        M, N = imR.shape
        p0   = (1., N/2, 50.)

        # Define the plume section as the 2*vWindowOffset+1 (~10) rows centred 
        # about the current location.  Section is the sum over these rows.
        row = imR[np.uint16(M/2)-vWindowOffset:np.uint16(M/2)+vWindowOffset, :]
        row = row.sum(axis=0)
        #row[np.uint16(N/2)+hWindowOffset:] = 0.
        row -= row[0]

        # Fit a Gaussian to the data and use this to define plume parameters
        X = np.arange(len(row))
        popt, pcov = curve_fit(gaussian_profile, X, row, p0)
        plumeWidth.append(popt[2])
        var_b.append(pcov[2][2])
        # Alternativly, apply the centroid method
        COM = centroidPosn(X, row, n=4)
        # Distance from centre of mass to the centre of the row, in
        # rotated coordinates
        d.append(popt[1] - (N/2)) # COM - (N/2) #
        var_d.append(d[-1]**2 * pcov[1][1])
        # Add the distance (i.e. residual) squared to the R2 value
        r2 += d[-1]**2
        # Do some geometry to find the "true location" of the plume centre
        # Should the distance from COM be added or subtracted from estimated
        # plume axis locus?
        th *= np.pi/180         # Convert back from degrees to radians
        trueLocn.append(np.float64(q) -   # Should this be a + or -?
                        np.multiply(d[-1], [-np.cos(th),np.sin(th)]))

        # ----------------------------------------------------------------- #
        # The remainder of this function is dedicated to plotting the
        # solutions.  Should probably incorporate a boolean handle that turns
        # this functionality on or off.
        # ----------------------------------------------------------------- #
        if plotting:
            # Plot the solution for visualisation purposes.
            # Update "true locations" on the base image
            ax00.plot(trueLocn[-1][0], trueLocn[-1][1], 'c+', ms=8, lw=3)
            
            # Rotate the image to show where current section is being calculated
            ax01.imshow(imR)
            ax01.axvline(N/2)
            ax01.axvline(N/2 - hWindowOffset, ls='--')
            ax01.axvline(N/2 + hWindowOffset, ls='--')
            ax01.axhline(M/2)
            
            # Plot the residual as a function of distance along plume axis
            ax10.plot(s, d[-1], '+', c='C0')
            ax10.set_title('$r^2$: %.4f' % r2)
            
            # Plot the section for the current rotation
            ax11.clear()
            ax11.plot(X, row, '-', c='C0', label='data')
            ax11.plot(X, gaussian_profile(X, *popt), 'r-', label='gaussian')
            ax11.axvline(COM, c='r', ls='--', lw=2, zorder=1)
            ax11.axvline(N/2, c='C0', ls='-', zorder=0)
            ax11.axvline(N/2 - hWindowOffset, c='C0', ls='--', zorder=0)
            ax11.axvline(N/2 + hWindowOffset, c='C0', ls='--', zorder=0)
            plt.pause(.5)  # Pause for a crude animation
        #plt.close()

    # Convert lists to numpy arrays for ease of manipulation
    var_b = np.array(var_b)
    var_d = np.array(var_d)
    d = np.array(d)
    if errors is not None:
        sig_trueLocn = np.sqrt(2 * errors[0]**2 + var_d + d**2 * sig_theta)
        return (np.array(trueLocn), np.abs(np.array(plumeWidth)),
                sig_trueLocn, np.sqrt(var_b))
    else:
        return np.array(trueLocn), np.abs(np.array(plumeWidth))


def pathFromSmoothedTheta(s, theta, snew, smoothing=0.): 
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
    

if __name__ == '__main__':
    plt.close('all')    # Clear pre-existing plots

    # Set a path variable according to which experiment is required
    #exptNo = 31
    #path   = './data/ExpPlumes_for_Dai/exp%02d/' % exptNo
    #fname  = path + 'gsplume.mat'
    path = '/home/ovsg/Documents/Thermographie/done/tiff/2019/mu sigma/'

    # Initial guesses.  Either load from file or make a fresh guess
    #with open(path + 'exp%02d_initGuess.json' % exptNo) as f:
    with open(path + '20190314_CSS_initGuess.json') as f:
        jsondata = json.load(f)
    p = np.array(jsondata['data'])
    
    fig, ax = plt.subplots()    
    try:
        axes = np.array([ax])
        ax, im, data, xexp, zexp, extent, (Ox, Oz) = openPlotExptImage(path,
                                                                       axes),
                                                                       #exptNo)
        leg  = ax.legend()
        leg.get_frame().set_facecolor('gray')                                   
    except TypeError:
        print('file %s doesn\'t seem to exist...skipping!' % fname)

    #p       = initialGuessAtAxis()  # Comment out if loading from file
    pPixels = p.copy() * scaleFactor
    pPixels[:,0] += Ox
    pPixels[:,1] -= Oz
    pPixels[:,1] *= -1

    thexp, sig_thexp = plumeAngle(p[:,0], p[:,1], errors=[1/scaleFactor]*2)
    _, bexp, sig_p, sig_bexp = trueLocationWidth(pPixels, data,
                                                 errors=[1/scaleFactor],
                                                 plotting=True)

    # Form the state variables that will be compared to model solution. 
    #p[:,0] = (p[:,0] - Ox) / scaleFactor
    #p[:,1] = (Oz - p[:,1]) / scaleFactor
    bexp     /= scaleFactor
    sig_bexp /= scaleFactor
    sig_p    /= scaleFactor
    sexp = distAlongPath(p[:,0], p[:,1])
    Vexp = np.array([bexp, thexp]).T
    sigV = np.array([sig_bexp, sig_thexp]).T

    # fname  = './data/ExpPlumes_for_Dai/GCTA_plumeData.xlsx'
    # writer = pandas.ExcelWriter(fname, engine='xlsxwriter') 
