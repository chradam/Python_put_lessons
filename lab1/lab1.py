import cv2
import numpy as np #for drawing

def zad1():
    cap = cv2.VideoCapture(0)
    key = ord('a')
    while key != ord('q'):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame comes here
        # Convert RGB image to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        # Detect edges on the blurred image
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)

        # Display the result of our processing
        cv2.imshow('result', img_edges)
        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(30)

    # When everything done, release the capture and destroy created windows
    cap.release()
    cv2.destroyAllWindows()

def zad2():
    # Load an color image in grayscale
    img = cv2.imread('photo1.jpg',cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)

    cv2.imwrite('photo1_new.jpg', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def zad3():
    # Load an color image
    img1 = cv2.imread('photo1.jpg')
    px1 = img1[220, 270]
    print("Rozmiary zdjęcia 1 i liczba kanałów" + str(img1.shape))
    print("Pixel value at [220, 270]: " + str(px1))

     # Load an image in grayscale
    img2 = cv2.imread('photo1.jpg', 0)
    px2 = img2[220, 270]
    print("Rozmiary zdjęcia 2 i liczba kanałów" + str(img2.shape))
    print("Pixel value at [220, 270]: " + str(px2))

    # kolorowe
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow('image1', img1)
    # szare
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.imshow('image2', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def zad4(): #dupliakt fragmanetu obrazu
    # Load an color image
    img = cv2.imread('photo1.jpg',cv2.IMREAD_COLOR)

    ball = img[380:540, 430:590]
    img[0:160, 0:160] = ball

    print("Rozmiary zdjęcia 1 i liczba kanalow" + str(img.shape))
    # kolorowe
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow('image1', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def zad5():
    # Load an color image
    img = cv2.imread('ac.png')
    b, g, r = cv2.split(img)
    cv2.imshow('image1', b)
    cv2.imshow('image2', g)
    cv2.imshow('image3', r)

    img_merge = cv2.merge((b, g, r))
    cv2.imshow('image4', img_merge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def zad6(): #frame flip capture
    cap = cv2.VideoCapture(0)
    # FourCC is a 4-byte code used to specify the video codec.
    # The list of available codes can be found in fourcc.org.
    # It is platform dependent. Following codecs works fine for me.

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output6.avi',fourcc,20.0,(640,480))
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            frame=cv2.flip(frame,0)
            out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def zad7(): #frame capture after space click
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output7.avi',fourcc,20.0,(640,480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def zad8():
    cap = cv2.VideoCapture('f.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        while cv2.waitKey(1) & 0xFF != ord(' '):
            pass
            # nothing happens when it executes
            # (pause - endless looping if the condition is still true)
    cap.release()
    cv2.destroyAllWindows()

def zad9(): #drawing rectangle
    img = cv2.imread('photo1.jpg')
    #top-left,top-right,color,thickness
    img = cv2.line(img,(790,55),(790,628),(255,255,0),15) #left
    img = cv2.line(img, (1390, 55), (1390, 628), (255, 255, 0), 15) #right
    img = cv2.line(img, (790, 55), (1390, 55), (255, 255, 0), 15)  # top
    img = cv2.line(img, (790, 628), (1390, 628), (255, 255, 0), 15)  # bottom

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Umim obrazy!', (730, 800), font, 3, (255, 255, 0), 12, cv2.LINE_AA)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    zad9()
