import cv2
import numpy as np
import math
import sys
import imutils
from skimage.color import rgb2yiq

# carrega imagem
# recebe: diretório de imagem
# retorna: imagem bgr
def load_image(image_file):
    # ler imagem
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return image

# binarização
# recebe: imagem bgr
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
# retorna: maior contorno da imagem, imagem com o contorno
def max_area_contour(mask, image):
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

    contour_image = cv2.drawContours(image, [cnt], -1, (255, 0, 0), 3)
    return cnt, contour_image

# aproximação do contorno da placa a um poligono
# recebe: contorno, imagem
# retorna: imagem binarizada com o contorno apenas
def approx_polygon(contour, image):
    # aproximar forma
    epsilon = 0.01*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    
    # desenhar contorno aproximado
    approx_contour = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(approx_contour, [approx], -1, (255, 0, 0), 3)
    
    return approx_contour

# aproximação do contorno da placa por preenchimento
# recebe: contorno, imagem
# retorna: imagem binarizada com o contorno apenas
def hull(contour, image):
    hull = cv2.convexHull(contour)

    # desenhar contorno preenchido
    approx_contour = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(approx_contour, [hull], -1, (255, 0, 0), 3)

    return approx_contour

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
# retorna: centroids dos blobs (harrisCorner), pontos dos cantos (subPix), imagem com os centroids e cantos
def get_corners(approx_contour, image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    corners = np.zeros((approx_contour.shape[0], approx_contour.shape[1]), np.uint8)

    dst = cv2.cornerHarris(approx_contour,20,3,0.04)
    dst = cv2.dilate(dst, None) # pra que isso?

    # Threshold for an optimal value, it may vary depending on the image.
    corners[dst>0.01*dst.max()] = 255

    # centroids dos blobs
    centroids = get_centroids(corners)

    # define the criteria to stop and refine the corners
    # melhorar resultado obtido com harrisCorner
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(11,11),(-1,-1),criteria)

    # desenhar centroids e corners
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

    # ordenar pela altura
    for j in range(3):
        for i in range(3):
            if points[i][0] > points[i+1][0]:
                aux_0 = points[i][0]
                aux_1 = points[i][1]
                points[i][0] = points[i+1][0]
                points[i][1] = points[i+1][1]
                points[i+1][0] = aux_0
                points[i+1][1] = aux_1
                #print(str(points))
                #np.delete(points, i)
                #np.append(i, points[i+1])

    if(points[0][1] > points[1][1]):
        top_left = points[1]
        top_right = points[0]
    else:
        top_left = points[0]
        top_right = points[1]
    if(points[2][1] > points[3][1]):
        bottom_left = points[3]
        bottom_right = points[2]
    else:
        bottom_left = points[2]
        bottom_right = points[3]
    #print("pontos ordenados: " + str(points))    

    pts = np.float32([top_left, bottom_left, top_right, bottom_right])
    #print(str(pts))
    return pts

# transforma a imagem geometricamente
# recebe: pontos, imagem
# retorna: imagem transformada
def set_perspective(points, image):
    # top_left, bottom_left, top_right, bottom_right
    # w = média das larguras
    # h = média das alturas
    w = int((points[2][1]-points[0][1] + points[3][1]-points[1][1])/2)
    h = int((points[1][0]-points[0][0] + points[3][0]-points[2][0])/2)
    #print(h)
    #print(w)

    points_1 = np.float32(points)
    points_2 = np.float32([[0,0],[h,0],[0,w],[h,w]])

    M = cv2.getPerspectiveTransform(points_1, points_2)

    dst = cv2.warpPerspective(image,M,(h,w))

    return dst

# detecta a placa
# recebe: diretório da imagem
# retorna: imagem contendo apenas a placa
def find_object(image_file):
    # carregar imagem
    image = load_image(image_file)
    cv2.imwrite("../results/image_"+image_file[10:], image)
    #cv2.namedWindow("image "+image_file[10:], cv2.WINDOW_NORMAL)
    #cv2.imshow("image "+image_file[10:], image)

    # binarização da imagem
    mask = thresholding(image)
    cv2.imwrite("../results/thresholding_"+image_file[10:], mask)
    #cv2.namedWindow("thresholding "+image_file[10:], cv2.WINDOW_NORMAL)
    #cv2.imshow("thresholding "+image_file[10:], mask)

    # maior contorno (placa)
    contour, image_contour = max_area_contour(mask, image.copy())
    cv2.imwrite("../results/contour_"+image_file[10:], image_contour)
    #cv2.namedWindow("contour "+image_file[10:], cv2.WINDOW_NORMAL)
    #cv2.imshow("contour "+image_file[10:], image_contour)

    # aproximação da forma
    approx_contour = hull(contour, image)
    cv2.imwrite("../results/approx_contour_"+image_file[10:], approx_contour)
    #cv2.namedWindow("approx. contour "+image_file[10:], cv2.WINDOW_NORMAL)
    #cv2.imshow("approx. contour "+image_file[10:], approx_contour)

    # detectar cantos
    centroids, corners, image_corners = get_corners(approx_contour, image.copy())
    cv2.imwrite("../results/corners_"+image_file[10:], image_corners)
    #cv2.namedWindow("corners "+image_file[10:], cv2.WINDOW_NORMAL)
    #cv2.imshow("corners "+image_file[10:], image_corners)

    # perspective
    # top_left, bottom_left, top_right, bottom_right
    points = order_points(corners)
    perspective = set_perspective(points, image.copy())
    cv2.imwrite("../results/perspective_"+image_file[10:], perspective)
    #cv2.namedWindow("perspective "+image_file[10:], cv2.WINDOW_NORMAL)
    #cv2.imshow("perspective "+image_file[10:], perspective)

# divide a imagem em várias plaquinhas menores e desenha círculos onde deveriam haver soldas pegando os pontos de um arquivo .txt
# recebe: imagem
# retorna: imagem com linhas delimitando as plaquinhas
def division(image_file):
    # arquivo .txt com a posição das soldas (porcentagem em relação ao tamanho da placa)
    file = open("new.txt", "r")
    soldas = []
    for line in file:
        line = line.split()
        soldas.append([float(line[0]), float(line[1])])
    file.close()
    soldas = np.asarray(soldas)
    #print(soldas)

    # imagem
    image = load_image(image_file)

    # partes do início e fim que devem ser cortadas pq não são placas
    dh_cut = image.shape[0]*0.04
    dw_cut = image.shape[1]*0.017

    # tamanho de cada placa tirando as partes que devem ser cortadas pq não tem plaquinha
    dh = (image.shape[0] - 2*dh_cut)/2
    dw = (image.shape[1] - 2*dw_cut)/10
    #print(dh)
    #print(dw)

    # pontos de referência pra desenhar as soldas (superior esquerdo de cada placa)
    pontos = []
    for i in range(10):
        pontos.append([i*dw+dw_cut, dh_cut])
    for i in range(10):
        pontos.append([i*dw+dw_cut, dh+dh_cut])
    #print(pontos)

    # distancia das soldas do ponto de referência
    soldas *= np.asarray([dw, dh])

    # desenhar soldas
    for ponto in pontos:
        for solda in soldas:
            #d = 0.04
            #print(ponto[0])
            #print(ponto[1])
            #print(solda[0])
            #print(solda[1])
            cv2.circle(image, center=(int(ponto[0]+solda[0]),int(ponto[1]+solda[1])), radius=int(dw*0.05), color=(255,255,255), thickness=1)
            #cv2.rectangle(image, pt1=(int(ponto[0]+solda[0]-dw*d),int(ponto[1]+solda[1]-dw*d)), pt2=(int(ponto[0]+solda[0]+dw*d),int(ponto[1]+solda[1]+dw*d)), color=(255,255,255), thickness=1)

    # desenhar as linhas que dividem as plaquinhas
    for i in range(3):
        cv2.line(image, (0, int(i*dh+dh_cut)), (image.shape[1], int(i*dh+dh_cut)), (255,255,255), 1)

    for i in range(11):
        cv2.line(image, (int(i*dw+dw_cut), 0), (int(i*dw+dw_cut), image.shape[0]), (255,255,255), 1)

    return image

def main():
    print("Processamento de imagens de PCB.")
    print("Processado com:")
    print("Python:", sys.version)
    print("OpenCV2: ", cv2.__version__)
    print("NumPy: ", np.__version__)

    # detectar somente a placa do resto da imagem
    images_file = ["../images/PCB_01_ilumin_03.jpg", "../images/PCB_01_ilumin_06.jpg"]
    for image_file in images_file:
        find_object(image_file)

    # divisão da imagem em plaquinhas menores
    images_file = ["../results/perspective_PCB_01_ilumin_03.jpg", "../results/perspective_PCB_01_ilumin_06.jpg"]
    for image_file in images_file:
        divided_rgb = division(image_file)
        cv2.imwrite("../results/divided_rgb_"+image_file[23:], divided_rgb)
        cv2.namedWindow("divided "+image_file, cv2.WINDOW_NORMAL)
        cv2.imshow("divided "+image_file, divided_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

main()