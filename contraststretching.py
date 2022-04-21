
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Image Negative script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/scene2.jpg')
    parser.add_argument('--min', type = int, help = "Minimum value of grayscale level", default = 0)
    parser.add_argument('--max', type = int, help = "Maximum value of grayscale level", default = 255)
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    hist = cv.calcHist([img],[0],None,[256],[0,256])
    Stretched = contraststretch(img, args.min/2, args.max/2)
    histStretched = cv.calcHist([Stretched],[0],None,[256],[0,256])

    plt.figure("Contrast Strectching")
    plt.subplot(2,3,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,3,4),plt.plot(hist)
    plt.title(f'Histogram')
    plt.yticks([])

    plt.subplot(2,3,2),plt.imshow(Stretched,'gray',vmin=0,vmax=255)
    plt.title(f'{int(args.min/2)} to {int(args.max/2)}')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,3,5),plt.plot(histStretched)
    plt.title(f'Histogram')
    plt.yticks([])

    Stretched = contraststretch(img, args.min, args.max)
    histStretched = cv.calcHist([Stretched],[0],None,[256],[0,256])

    plt.subplot(2,3,3),plt.imshow(Stretched,'gray',vmin=0,vmax=255)
    plt.title(f'{args.min} to {args.max}')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,3,6),plt.plot(histStretched)
    plt.title(f'Histogram')
    plt.yticks([])

    plt.show()

def contraststretch(image, min, max):
    norm = np.zeros(image.shape)
    return cv.normalize(image, norm, min, max, cv.NORM_MINMAX)


if __name__ == '__main__':
    main()