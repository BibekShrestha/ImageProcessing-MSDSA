
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "log correction Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/scene2.jpg')
    parser.add_argument('-c', type = int, help = "constant c of log correction equation", default = 1)
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    hist = cv.calcHist([img],[0],None,[256],[0,256])
    logtransformed = logtransform(img, args.c)
    histlogtransformed = cv.calcHist([logtransformed],[0],None,[256],[0,256])

    plt.figure("Log Transformation")
    plt.subplot(2,2,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,4),plt.plot(hist)
    plt.title(f'Histogram')
    plt.yticks([])

    plt.subplot(2,2,2),plt.imshow(logtransformed,'gray',vmin=0,vmax=255)
    plt.title(f'c = {args.c}')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,3),plt.plot(histlogtransformed)
    plt.title(f'Histogram')
    plt.yticks([])

    plt.show()

def logtransform(image, c):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(c*np.log10(1 + i/255)*255, 0, 255)
    return cv.LUT(image, lookUpTable)


if __name__ == '__main__':
    main()