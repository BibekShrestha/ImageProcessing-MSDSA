import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from spnoise import spnoise
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Thresholding script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/panda1.jpg')
    parser.add_argument('-s', '--masksize', type = int, help = 'size of mask used', default = 3)
    parser.add_argument('-b', '--boostvalue', type = int, help = 'boost value', default = 1.1)
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    img = cv.imread(args.image,0)
    filtered = cv.Laplacian(img, -1, args.masksize);
    A = args.boostvalue if args.boostvalue >= 1 else 1.5
    enhanced = args.boostvalue  * img + filtered
    
    plt.figure("Laplacian Filtering")
    plt.subplot(3,1,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(3,1,2),plt.imshow(filtered,'gray',vmin=0,vmax=255)
    plt.title('Laplacian High pass')
    plt.xticks([]),plt.yticks([])

    plt.subplot(3,1,3),plt.imshow(enhanced,'gray',vmin=0,vmax=255)
    plt.title('High Boost')
    plt.xticks([]),plt.yticks([])

    plt.show()

def filter(image,masksize):
    mask = np.ones((masksize,masksize),np.float32)/(masksize*masksize)
    return cv.filter2D(image, -1, mask)

if __name__ == '__main__':
    main()