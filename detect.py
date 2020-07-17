import cv2
import numpy as np
import math
import sys
import imutils

# carrega imagem
# recebe: diretório de imagem
# retorna: imagem
def load_image(image_file):
    # ler imagem
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return image

# binarização
# recebe: imagem
# retorna: imagem binarizada
def thresholding(image):
    # transforma imagem rgb em hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # binarização da imagem
    green_lower = (60, 60, 20)
    green_upper = (100, 255, 140)
    mask = cv2.inRange(hsv, green_lower, green_upper)

    return mask

# contorno da placa
# recebe: imagem binarizada
# retorna: maior contorno da imagem
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

# aproximação do contorno da placa a um poligono
# recebe: contorno, imagem
# retorna: imagem binarizada com o contorno apenas
def approx_polygon(contour, image):
    # aproximar forma
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    
    # desenhar contorno aproximado
    contour_approx = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(contour_approx, [approx], -1, (255, 0, 0), 3)
    
    return contour_approx

# aproximação do contorno da placa por preenchimento
# recebe: contorno, imagem
# retorna: imagem binarizada com o contorno apenas
def hull(contour, image):
    hull = cv2.convexHull(contour)

    # desenhar contorno aproximado
    contour_approx = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(contour_approx, [hull], -1, (255, 0, 0), 3)

    return contour_approx

# centroids
# recebe: imagem binarizada
# retorna: centroids dos blobs
def get_centroids(corners):
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

# cantos da imagem
# recebe: contorno aproximado, imagem
# retorna: pontos do centroid (harrisCorner), pontos dos cantos (subPix), imagem com os dois pontos
def get_corners(contour_approx, image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    corners = np.zeros((contour_approx.shape[0], contour_approx.shape[1]), np.uint8)

    dst = cv2.cornerHarris(contour_approx,20,3,0.04)
    dst = cv2.dilate(dst, None) # pra que isso?

    # Threshold for an optimal value, it may vary depending on the image.
    corners[dst>0.01*dst.max()] = 255

    centroids = get_centroids(corners)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(11,11),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    image[res[:,1],res[:,0]] = [0,0,255]
    image[res[:,3],res[:,2]] = [0,255,0]

    return centroids, corners, image

# ordena 4 pontos
# recebe: pontos
# retorna: pontos ordenados
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

# transforma a imagem geometricamente
# recebe: pontos, imagem
# retorna: imagem transformada
def set_perspective(points, image):
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
    contour_approx = hull(contour, image)

    # detectar cantos
    centroid, corners, image_corners = get_corners(contour_approx, image.copy())

    # top_left, bottom_left, top_right, bottom_right
    points = order_points(corners)

    perspective = set_perspective(points, image.copy())
    
    # perspective
    cv2.namedWindow(image_file, cv2.WINDOW_NORMAL)
    cv2.imshow(image_file, perspective)

    # cantos
    cv2.namedWindow("cantos "+image_file, cv2.WINDOW_NORMAL)
    cv2.imshow("cantos "+image_file, image_corners)

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