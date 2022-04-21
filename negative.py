import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Image Negative script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/panda1.jpg')
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    negated = negative(img,255);
    
    plt.figure("Negative ")
    plt.subplot(2,1,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,1,2),plt.imshow(negated,'gray',vmin=0,vmax=255)
    plt.title('Negative Image')
    plt.xticks([]),plt.yticks([])

    plt.show()

def negative(image, max = 255):
    return (max - image).astype('uint8')

if __name__ == '__main__':
    main()