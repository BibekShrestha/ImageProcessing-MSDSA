
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Image Negative script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/scene2.jpg')
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    hist = cv.calcHist([img],[0],None,[256],[0,256])
    EqImage = histeq(img)
    histEqImage = cv.calcHist([EqImage],[0],None,[256],[0,256])

    plt.figure("Histogram Equalization")
    plt.subplot(2,2,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,3),plt.plot(hist)
    plt.title(f'Histogram')
    plt.yticks([])

    plt.subplot(2,2,2),plt.imshow(EqImage,'gray',vmin=0,vmax=255)
    plt.title(f'Histogram Equalized Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,4),plt.plot(histEqImage)
    plt.title(f'Histogram')
    plt.yticks([])

    plt.show()

def histeq(image,bit):
    return (image & (1 << bit)).astype('bool').astype('uint8')*255

def histeq(image):
    return cv.equalizeHist(image)
    


if __name__ == '__main__':
    main()