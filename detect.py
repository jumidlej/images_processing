import cv2
import numpy as np
import math
import sys
import imutils

def load_image(image_file):
    # ler imagem
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return image

def thresholding(image):
    # transforma imagem rgb em hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # binarização da imagem
    green_lower = (60, 60, 20)
    green_upper = (100, 255, 140)
    mask = cv2.inRange(hsv, green_lower, green_upper)

    return mask

def max_area_contour(mask):
    # contornos
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # maior contorno
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for contour in contours:
        if cv2.contourArea(contour) > max_area:
            cnt = contour
            max_area = cv2.contourArea(cnt)

    return cnt

def approx_polygon(contour):
    # aproximar forma
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    return approx

def hull(contour):
    hull = cv2.convexHull(contour)
    return hull

def get_corners(contour_approx):
    corners = np.zeros((contour_approx.shape[0], contour_approx.shape[1]), np.uint8)

    dst = cv2.cornerHarris(contour_approx,20,3,0.04)
    # Threshold for an optimal value, it may vary depending on the image.
    corners[dst>0.01*dst.max()] = 255

    return corners

def get_points(corners):
    contours = cv2.findContours(corners, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    points = []

    # centroid
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        points.append((cx, cy))

    return points

def order_points(points):
    top_left = [0, 0]
    top_right = [0, 0]
    bottom_right = [0, 0]
    bottom_left = [0, 0]

    max_h_1 = [0, 0]
    max_h_2 = [0, 0]
    min_h_1 = [6000, 6000]
    min_h_2 = [6000, 6000]
    for i in range(4):
        if points[i][0] > max_h_1[0]:
            max_h_1 = points[i]
        elif points[i][0] > max_h_2[0]:
            max_h_2 = points[i]
        if points[i][0] < min_h_1[0]:
            min_h_1 = points[i]
        elif points[i][0] < min_h_2[0]:
            min_h_2 = points[i]
        if max_h_1[0] < max_h_2[0]:
            aux = max_h_1
            max_h_1 = max_h_2
            max_h_2 = aux
        if min_h_1[0] > min_h_2[0]:
            aux = min_h_1
            min_h_1 = min_h_2
            min_h_2 = aux

    if max_h_1[1] >= max_h_2[1]:
        bottom_right = max_h_1
        bottom_left = max_h_2
    else:  
        bottom_right = max_h_2
        bottom_left = max_h_1

    if min_h_1[1] >= min_h_2[1]:
        top_right = min_h_1
        top_left = min_h_2
    else:  
        top_right = min_h_2
        top_left = min_h_1

    pts = np.uint32([top_left, bottom_left, top_right, bottom_right])
    return pts

def set_perspective(image, points):
    h = 3100
    w = 1400

    points_1 = np.float32(points)
    points_2 = np.float32([[0,0],[h,0],[0,w],[h,w]])

    M = cv2.getPerspectiveTransform(points_1, points_2)

    dst = cv2.warpPerspective(image,M,(h,w))

    return dst

def find_object(image_file):
    # carregar imagem
    image = load_image(image_file)

    # binarização da imagem
    mask = thresholding(image)

    # maior contorno (placa)
    contour = max_area_contour(mask)
    contour_image = cv2.drawContours(image.copy(), [contour], -1, (255, 0, 0), 3)

    # aproximação da forma
    # approx = approx_polygon(contour)
    approx = hull(contour)

    # desenhar contorno aproximado
    contour_approx = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(contour_approx, [approx], -1, (255, 0, 0), 3)

    # detectar cantos
    corners = get_corners(contour_approx)
    
    points = get_points(corners)
    #print(str(points))

    points = order_points(points)
    #print(str(points))

    # desenhar pontos na imagem
    image_points = image.copy()
    for point in points:
        cv2.circle(image_points, (point[0], point[1]), 10, (255, 0, 255))

    perspective = set_perspective(image.copy(), points)
    
    # perspective
    cv2.namedWindow(image_file, cv2.WINDOW_NORMAL)
    cv2.imshow(image_file, perspective)

    '''
    # contorno
    cv2.namedWindow("contorno placa", cv2.WINDOW_NORMAL)
    cv2.imshow("contorno placa", contour_image)

    # pontos
    cv2.namedWindow("pontos", cv2.WINDOW_NORMAL)
    cv2.imshow("pontos", image_points)
    '''

def main():
    print("Processamento de imagens de PCB.")
    print("Processado com:")
    print("Python:", sys.version)
    print("OpenCV2: ", cv2.__version__)
    print("NumPy: ", np.__version__)

    images_file = ["../images/PCB_01_ilumin_03.jpg", "../images/PCB_01_ilumin_06.jpg"]
    for image_file in images_file:
        find_object(image_file)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0

main()