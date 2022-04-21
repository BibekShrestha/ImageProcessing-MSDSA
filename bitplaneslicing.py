usage = '''
'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def parse_arguments():
    bitchoices = [0,1,2,3,4,5,6,7]
    parser = argparse.ArgumentParser(description = "Image Negative script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', type = str, help = 'Name of image to load', default = 'assets/panda1.jpg')
    parser.add_argument('-b1', type = int, choices = bitchoices, help = 'Bit number choice 1', default = '7')
    parser.add_argument('-b2', type = int, choices = bitchoices, help = 'Bit number choice 2', default = '5')
    parser.add_argument('-b3', type = int, choices = bitchoices, help = 'Bit number choice 3', default = '3')
    return parser.parse_args()

def main():
    args = parse_arguments()
    img = cv.imread(args.image,0)

    b1 = bitget(img,args.b1)
    b2 = bitget(img,args.b2)
    b3 = bitget(img,args.b3)

    plt.figure("Bit Plane Slicing")
    plt.subplot(2,2,1),plt.imshow(img,'gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,2),plt.imshow(b1,'gray',vmin=0,vmax=255)
    plt.title(f'Bit {args.b1} Image.')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,3),plt.imshow(b2,'gray',vmin=0,vmax=255)
    plt.title(f'Bit {args.b2} Image.')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,4),plt.imshow(b3,'gray',vmin=0,vmax=255)
    plt.title(f'Bit {args.b3} Image.')
    plt.xticks([]),plt.yticks([])

    plt.show()

def bitget(image,bit):
    return (image & (1 << bit)).astype('bool').astype('uint8')*255

if __name__ == '__main__':
    main()