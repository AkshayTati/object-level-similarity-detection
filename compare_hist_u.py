# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="Path to the directory of images")
# args = vars(ap.parse_args())
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}


# loop over the image paths
def get_images(path):
    for imagePath in glob.glob(path + "\\*.jpg"):
        # extract the image filename (assumed to be unique) and
        # load the image, updating the images dictionary
        filename = imagePath[imagePath.rfind("\\") + 1:]
        image = cv2.imread(imagePath)
        images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        index[filename] = hist
    return index


# METHOD #1: UTILIZING OPENCV
# initialize OpenCV methods for histogram comparison
# OPENCV_METHODS = (
#     ("Correlation", cv2.HISTCMP_CORREL),
#     ("Chi-Squared", cv2.HISTCMP_CHISQR),
#     ("Intersection", cv2.HISTCMP_INTERSECT),
#     ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))


def get_hist_metrics(method, index, baseImage):
    results = {}

    reverse = False
    # if we are using the correlation or intersection
    # method, then sort the results in reverse order
    if method in (cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT):
        reverse = True

    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(index[baseImage], hist, method)
        results[k] = d
    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
    # show the query image
    # fig = plt.figure("Query")
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(images[baseImage])
    # plt.axis("off")
    # initialize the results figure

    return results

def plot_result(results, methodName, baseImage):
    fig = plt.figure("Results: %s" % (methodName))
    fig.suptitle(methodName, fontsize=20)
    # loop over the results
    for (i, (v, k)) in enumerate(results):
        # show the result
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")
    for (i, (v, k)) in enumerate(results):
        if v<=15.0 and k!=baseImage:
            plt.figtext(0.5, 0.01, "Similar", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        elif k!=baseImage:
            plt.figtext(0.5, 0.01, "Not Similar", ha="center", fontsize=18,
                        bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    plt.show()


def main():
    images_ = get_images("detected")
    results = get_hist_metrics(cv2.HISTCMP_CORREL, images_, "object0.jpg")
    plot_result(results, "Chi Squared", "object0.jpg")


if __name__ == "__main__":
    main()
