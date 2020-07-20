# Importing needed library
import cv2
import imutils

# Preparing Track Bars
# Defining empty function
def do_nothing(x):
    pass

# Giving name to the window with Track Bars
# And specifying that window is resizable
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

# Defining Track Bars for convenient process of choosing colours
# For minimum range
cv2.createTrackbar('min', 'Track Bars', 0, 255, do_nothing)

# For preprocessing
cv2.createTrackbar('erode', 'Track Bars', 0, 10, do_nothing)
cv2.createTrackbar('dilate', 'Track Bars', 0, 10, do_nothing)

# For maximum range
cv2.createTrackbar('max', 'Track Bars', 0, 255, do_nothing)

# Adaptative
cv2.createTrackbar('k', 'Track Bars', 5, 200, do_nothing)
cv2.createTrackbar('c', 'Track Bars', -0, 200, do_nothing)

# Reading image with OpenCV library
# In this way image is opened already as numpy array
# WARNING! OpenCV by default reads images in BGR format
image = 'normalized3_equalized_yiq_PCB_01_ilumin_06.jpg'
image_BGR = cv2.imread('../results/'+image)
# Resizing image in order to use smaller windows
# image_BGR = imutils.resize(image_BGR, width=800)

# Showing Original Image
# Giving name to the window with Original Image
# And specifying that window is resizable
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', imutils.resize(image_BGR, width=800))

# Converting Original Image to HSV
image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

# Defining loop for choosing right Colours for the Mask
while True:
    # Defining variables for saving values of the Track Bars
    # For minimum range
    min_gray = cv2.getTrackbarPos('min', 'Track Bars')

    # For maximum range
    max_gray = cv2.getTrackbarPos('max', 'Track Bars')

    # Implementing Mask with chosen colours from Track Bars to HSV Image
    # Defining lower bounds and upper bounds for thresholding
    erode = cv2.getTrackbarPos('erode', 'Track Bars')
    dilate = cv2.getTrackbarPos('dilate', 'Track Bars')

    k = cv2.getTrackbarPos('k', 'Track Bars')
    c = cv2.getTrackbarPos('c', 'Track Bars')

    if k % 2 == 0:
        k += 1
    if c > 100:
        c -= 100
        c -= 2*c

    #ret,th = cv2.threshold(image,min_gray,max_gray,cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,k, -c)
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,k, -c)

    #th = cv2.erode(th, None, iterations=erode)
    #th = cv2.dilate(th, None, iterations=dilate)
    #th2 = cv2.erode(th2, None, iterations=erode)
    #th2 = cv2.dilate(th2, None, iterations=dilate)
    th3 = cv2.erode(th3, None, iterations=erode)
    th3 = cv2.dilate(th3, None, iterations=dilate)

    # Showing Binary Image with implemented Mask
    # Giving name to the window with Mask
    # And specifying that window is resizable
    #cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
    #cv2.imshow('Thresh', th)
    #cv2.namedWindow('Gauss', cv2.WINDOW_NORMAL)
    #cv2.imshow('Gauss', th2)
    cv2.namedWindow('Mean', cv2.WINDOW_NORMAL)
    cv2.imshow('Mean', th3)


    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()

# Printing final chosen Mask numbers
print('min, max = {0}, {1}'.format(min_gray, max_gray))
print('k, c = {0}, {1}'.format(k, c))
print('erode, dilate = {0}, {1}'.format(erode, dilate))
# Printing final chosen Mask numbers
'''
arquivo = open("parametros.txt", "a")
arquivo.write(image+'\n')
arquivo.write('min_blue, min_green, min_red = {0}, {1}, {2}'.format(min_blue, min_green, min_red)+'\n')
arquivo.write('max_blue, max_green, max_red = {0}, {1}, {2}'.format(max_blue, max_green, max_red)+'\n')
arquivo.write('erode, dilate = {0}, {1}'.format(erode, dilate)+'\n')
arquivo.close()
'''