""" 
From skimage recipe book 

http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
"""

from skimage import exposure, img_as_float

import matplotlib.pyplot as plt
import numpy as np 


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


if __name__ == "__main__":
    # load an image...
    img = plt.imread('./Carazzo_MovieS1_average.jpg')
    # ...then extract its blue channel only
    img = img[:,:,2]

    bins = 256
    
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    # It is reasonable to assume that the background occupies most of the image.
    # Thus if the mean intensity is skewed towards black, the background is 
    # likely to be dark with a light plume and vice-versa.  Ensure that the 
    # plume is lighter than the background as this will be important when later
    # fitting a gaussian to the intensity profile (bright pixels represent the 
    # "peak".
    if img.mean() > 127:
        img = np.invert(img)

    # Display results
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 5))
    ax_cdf = ax1.twinx()

    # Display image
    ax0.imshow(img, cmap=plt.cm.gray)
    ax0.set_axis_off()
    ax0.set_adjustable('box')

    # Display histogram
    ax1.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax1.set_xlabel('Pixel intensity')
    #ax1.set_xlim(0, 1)
    ax1.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])
    
    ax0.set_title('Contrast stretching')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

    ## NOTE - Contrast stretching seems to be the most reliable method.
