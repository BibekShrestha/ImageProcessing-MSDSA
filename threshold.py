import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Thresholding script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/panda1.jpg')
    parser.add_argument('-t', '--threshold', type = int, help = 'Desired threshold value.', default = 127)
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    thresholded = threshold(img,args.threshold,255);

    plt.figure("Thresholding")
    plt.subplot(2,1,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,1,2),plt.imshow(thresholded,'gray',vmin=0,vmax=255)
    plt.title('Threshold = {}'.format(args.threshold))
    plt.xticks([]),plt.yticks([])

    plt.show()

def threshold(image, threshold, max = 255):
    return (image > threshold).astype('uint8')*max

if __name__ == '__main__':
    main()