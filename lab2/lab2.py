import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    print("Trackbar reporting for duty with value: "  + str(x))
def zad1():
    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)
    cv2.createTrackbar('G','image',0,255,nothing)
    cv2.createTrackbar('B','image',0,255,nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    #trackbarName,windowName,value,count,onChange
    cv2.createTrackbar(switch, 'image',0,1,nothing)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')

        if s == 0: #obraz czarny
            img[:] = 0
        else: #mix kolorów
            img[:] = [b,g,r]

    cv2.destroyAllWindows()
#types of thresholding:

#cv2.THRESH_BINARY
#cv2.THRESH_BINARY_INV
#cv2.THRESH_TRUNC
#cv2.THRESH_TOZERO
#cv2.THRESH_TOZERO_INV

def zad2():
    img = cv2.imread('photo1.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #(src,tresh,maxval,type,type,dst)
    #powyżej wartości 127 piksele są kolorowane na biało(255)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #na odwrót do poprzedniego
    ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        #subplot - 2 rzędy po 3 zdjęcia, 6 zdjęć z macierzy
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        #wartośći na osiach zdjęcia zostały ukryte
        plt.xticks([]), plt.yticks([])

    plt.show()

def zad3(): # threshold + truckbar
    photo = cv2.imread('photo1.jpg', 0)
    photo = cv2.resize(photo,(0,0),fx=0.3, fy=0.3)
    #cv2.namedWindow('bar', cv2.WINDOW_NORMAL)
    cv2.namedWindow('bar')
    cv2.createTrackbar('V', 'bar', 0, 255, nothing)
    cv2.createTrackbar('T', 'bar', 0, 4, nothing)
    tresholds = [cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV]
    while (1):
        v = cv2.getTrackbarPos('V', 'bar')
        t = cv2.getTrackbarPos('T', 'bar')
        ret, thresh1 = cv2.threshold(photo, v, 255, t)
        cv2.imshow('bar', thresh1)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
def zad4(): #resize
    photo = cv2.imread('qr.jpg', 0)
    s1 = cv2.resize(photo, (0, 0), fx=1.75, fy=1.75,interpolation=cv2.INTER_LINEAR) #domyślnie
    s2 = cv2.resize(photo, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_NEAREST)
    s3 = cv2.resize(photo, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_AREA)
    s4 = cv2.resize(photo, (0, 0), fx=1.75, fy=1.75, interpolation=cv2.INTER_LANCZOS4) #najlepszy
    cv2.namedWindow('s1')
    cv2.namedWindow('s2')
    cv2.namedWindow('s3')
    cv2.namedWindow('s4')
    cv2.imshow('s1', s1)
    cv2.imshow('s2', s2)
    cv2.imshow('s3', s3)
    cv2.imshow('s4', s4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def zad5(): #obrazy jako macierze
    pass
if __name__ == "__main__":
    #zad1()
    #zad2()
    #zad3()
    #zad4()
    zad5()