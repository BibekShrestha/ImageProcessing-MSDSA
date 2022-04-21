import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from spnoise import spnoise
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Thresholding script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/panda1.jpg')
    parser.add_argument('-s', '--masksize', type = int, help = 'size of mask used', default = 3)
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    img = cv.imread(args.image,0)
    filtered = filter(img,args.masksize);

    plt.figure("Spatial Filter")
    plt.subplot(2,2,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,3),plt.imshow(filtered,'gray',vmin=0,vmax=255)
    plt.title('masksize = {}'.format(args.masksize))
    plt.xticks([]),plt.yticks([])

    noisyimage = spnoise(img)
    noisyfiltered = filter(noisyimage, args.masksize)

    plt.subplot(2,2,2),plt.imshow(noisyimage,'gray',vmin=0,vmax=255)
    plt.title('SP Noisy Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,4),plt.imshow(noisyfiltered,'gray',vmin=0,vmax=255)
    plt.title('masksize = {}'.format(args.masksize))
    plt.xticks([]),plt.yticks([])

    plt.show()

def filter(image,masksize):
    mask = np.ones((masksize,masksize),np.float32)/(masksize*masksize)
    return cv.filter2D(image, -1, mask)

if __name__ == '__main__':
    main()