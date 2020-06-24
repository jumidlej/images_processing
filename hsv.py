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
cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_red', 'Track Bars', 0, 255, do_nothing)

# For preprocessing
cv2.createTrackbar('erode', 'Track Bars', 0, 10, do_nothing)
cv2.createTrackbar('dilate', 'Track Bars', 0, 10, do_nothing)

# For maximum range
cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_red', 'Track Bars', 0, 255, do_nothing)

# Reading image with OpenCV library
# In this way image is opened already as numpy array
# WARNING! OpenCV by default reads images in BGR format
image = 'PCB_01_ilumin_06.jpg'
image_BGR = cv2.imread('../imagens/'+image)
# Resizing image in order to use smaller windows
# image_BGR = imutils.resize(image_BGR, width=800)

# Showing Original Image
# Giving name to the window with Original Image
# And specifying that window is resizable
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', imutils.resize(image_BGR, width=800))

# Converting Original Image to HSV
image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

# Showing HSV Image
# Giving name to the window with HSV Image
# And specifying that window is resizable
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
cv2.imshow('HSV Image', imutils.resize(image_HSV, width=800))


# Defining loop for choosing right Colours for the Mask
while True:
    # Defining variables for saving values of the Track Bars
    # For minimum range
    min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
    min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
    min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

    # For maximum range
    max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
    max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
    max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

    # Implementing Mask with chosen colours from Track Bars to HSV Image
    # Defining lower bounds and upper bounds for thresholding
    erode = cv2.getTrackbarPos('erode', 'Track Bars')
    dilate = cv2.getTrackbarPos('dilate', 'Track Bars')

    mask = cv2.inRange(image_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    mask = cv2.erode(mask, None, iterations=erode)
    mask = cv2.dilate(mask, None, iterations=dilate)

    # Showing Binary Image with implemented Mask
    # Giving name to the window with Mask
    # And specifying that window is resizable
    cv2.namedWindow('Binary Image with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image with Mask',mask)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()

# Printing final chosen Mask numbers
print('min_blue, min_green, min_red = {0}, {1}, {2}'.format(min_blue, min_green, min_red))
print('max_blue, max_green, max_red = {0}, {1}, {2}'.format(max_blue, max_green, max_red))
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