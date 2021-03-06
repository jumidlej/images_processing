import cv2
import numpy as np
import math
import sys
import imutils
from skimage.color import rgb2yiq

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

    # desenhar contorno aproximado
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
# retorna: pontos do centroid (harrisCorner), pontos dos cantos (subPix), imagem com os dois pontos
def get_corners(approx_contour, image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    corners = np.zeros((approx_contour.shape[0], approx_contour.shape[1]), np.uint8)

    dst = cv2.cornerHarris(approx_contour,20,3,0.04)
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

# divide a imagem em várias plaquinhas menores
# recebe: imagem
# retorna: imagem com linhas delimitando as plaquinhas
def division(image_file):
    #edge = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    #edge = image
    #edge[30:-30,30:-30,:] = (0, 0, 0)
    #if len(image.shape) == 2:
    #    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    file = open("new.txt", "r")
    soldas = []
    for line in file:
        line = line.split()
        soldas.append([float(line[0]), float(line[1])])
    file.close()
    soldas = np.asarray(soldas)
    #print(soldas)

    image = load_image(image_file)

    dh_cut = image.shape[0]*0.04
    dw_cut = image.shape[1]*0.017

    dh = (image.shape[0] - 2*dh_cut)/2
    dw = (image.shape[1] - 2*dw_cut)/10
    #print(dh)
    #print(dw)

    pontos = []
    for i in range(10):
        pontos.append([int(i*dw+dw_cut), int(dh_cut)])
    for i in range(10):
        pontos.append([int(i*dw+dw_cut), int(dh+dh_cut)])
    #print(pontos)
    # desenhar soldas
    soldas *= np.asarray([dw, dh])
    for ponto in pontos:
        for solda in soldas:
            d = 0.04
            #print(ponto[0])
            #print(ponto[1])
            #print(solda[0])
            #print(solda[1])
            #cv2.circle(image, center=(int(ponto[0]+solda[0]),int(ponto[1]+solda[1])), radius=int(dw*0.05), color=(255,255,255), thickness=1)
            cv2.rectangle(image, pt1=(int(ponto[0]+solda[0]-dw*d),int(ponto[1]+solda[1]-dw*d)), pt2=(int(ponto[0]+solda[0]+dw*d),int(ponto[1]+solda[1]+dw*d)), color=(255,255,255), thickness=1)

    for i in range(3):
        cv2.line(image, (0, int(i*dh+dh_cut)), (image.shape[1], int(i*dh+dh_cut)), (255,255,255), 1)

    for i in range(11):
        cv2.line(image, (int(i*dw+dw_cut), 0), (int(i*dw+dw_cut), image.shape[0]), (255,255,255), 1)

    return image

# equalização
# recebe: imagem da placa
# retorna: imagem equalizada
def equalization(image):
    #print(image.shape)
    if len(image.shape) == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    for i in range(image.shape[2]):
        image[:,:,i] = cv2.equalizeHist(image[:,:,i])
    
    return image

# normalização
# recebe: imagem da placa RGB
# retorna: imagem normalizada
def normalization(image, N=4):
    if len(image.shape) == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    h, w, d = image.shape
    #print(h, w, d)
    dstRGB = np.copy(image)
    for i in range(d):
        miu = np.log10(image[:, :, i].mean())
        C_00 = miu * math.sqrt(h * w)
        # C_00 = 0
        vis0 = np.zeros((h, w), np.float32)
        vis3 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = image[:h, :w, i]
        vis0 += 1
        c = 255 / np.log10(vis0.max())
        #print(c)
        vis0 = c * np.log10(vis0)
        # vis0_1 = cv2.normalize(vis0, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("Log", vis0_1)
        vis1 = cv2.dct(vis0)
        # cv2.imshow("DCT", vis1)
        for k in range(N):
            for j in range(i + 1):
                vis1[k - j, j] = C_00
        # cv2.imshow("DCT Normalize", vis1)

        vis2 = cv2.idct(vis1)
        # vis2 = cv2.normalize(vis2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("iDCT", vis2)
        vis3 = (1.02 ** vis2) - 1
        vis3 = cv2.normalize(vis3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dstRGB[:h, :w, i] = vis3[:h, :w]
    return dstRGB

def YIQ(image):
    image_yiq = rgb2yiq(image)
    image_yiq = np.around(image_yiq * 255, decimals=0)

    #print(image_yiq)
    #print(type(image_yiq))
    image_yiq = image_yiq.astype(np.uint8)
    #image_yiq = image_yiq.astype(np.float32)
    return image_yiq

def HSV(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(image_hsv)
    #print(type(image_hsv))
    return image_hsv

def BW(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(type(image_bw))
    return image_bw

def thresh(image):
    #ret,th1 = cv2.threshold(image,140,255,cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,-40)
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,-40)
    return th3

def main():
    print("Processamento de imagens de PCB.")
    print("Processado com:")
    print("Python:", sys.version)
    print("OpenCV2: ", cv2.__version__)
    print("NumPy: ", np.__version__)

    images_file = ["../images/PCB_01_ilumin_03.jpg", "../images/PCB_01_ilumin_06.jpg"]
    for image_file in images_file:
        find_object(image_file)

    '''
    images_file = ["../results/perspective_PCB_01_ilumin_03.jpg", "../results/perspective_PCB_01_ilumin_06.jpg"]
    for image_file in images_file:

        # sistemas de cores
        rgb = load_image(image_file)
        #rgb = rgb[10:-10,10:-10,:]
        cv2.imwrite("../results/rgb_"+image_file[23:], rgb)

        yiq = YIQ(rgb.copy())
        cv2.imwrite("../results/yiq_"+image_file[23:], yiq[:,:,0])

        hsv = HSV(rgb.copy())
        cv2.imwrite("../results/hsv_"+image_file[23:], hsv)

        bw = BW(rgb.copy())
        cv2.imwrite("../results/bw_"+image_file[23:], bw)

        # equalização sobre as imagens originais
        equalized_rgb = equalization(rgb.copy())
        cv2.imwrite("../results/equalized_rgb_"+image_file[23:], equalized_rgb)

        equalized_bw = equalization(bw.copy())
        cv2.imwrite("../results/equalized_bw_"+image_file[23:], equalized_bw)

        equalized_yiq = equalization(yiq.copy())
        cv2.imwrite("../results/equalized_yiq_"+image_file[23:], equalized_yiq[:, :, 0])

        equalized_hsv = equalization(hsv.copy())
        cv2.imwrite("../results/equalized_hsv_"+image_file[23:], equalized_hsv)

        # normalização sobre as imagens originais
        n = 3
        normalized_rgb = normalization(rgb.copy(), 5)
        cv2.imwrite("../results/normalized"+str(n)+"_rgb_"+image_file[23:], normalized_rgb)

        normalized_bw = normalization(bw.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_bw_"+image_file[23:], normalized_bw)

        normalized_yiq = normalization(yiq.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_yiq_"+image_file[23:], normalized_yiq[:, :, 0])

        normalized_hsv = normalization(hsv.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_hsv_"+image_file[23:], normalized_hsv)

        # equalização sobre imagens normalizadas
        normalized_equalized_rgb = equalization(normalized_rgb.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_equalized_rgb_"+image_file[23:], normalized_equalized_rgb)

        normalized_equalized_bw = equalization(normalized_bw.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_equalized_bw_"+image_file[23:], normalized_equalized_bw)

        normalized_equalized_yiq = equalization(normalized_yiq.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_equalized_yiq_"+image_file[23:], normalized_equalized_yiq[:, :, 0])

        normalized_equalized_hsv = equalization(normalized_hsv.copy())
        cv2.imwrite("../results/normalized"+str(n)+"_equalized_hsv_"+image_file[23:], normalized_equalized_hsv)

        # normalização sobre imagens equalizadas
        equalized_normalized_rgb = normalization(equalized_rgb.copy())
        cv2.imwrite("../results/equalized_normalized"+str(n)+"_rgb_"+image_file[23:], equalized_normalized_rgb)

        equalized_normalized_bw = normalization(equalized_bw.copy())
        cv2.imwrite("../results/equalized_normalized"+str(n)+"_bw_"+image_file[23:], equalized_normalized_bw)

        equalized_normalized_yiq = normalization(equalized_yiq.copy())
        cv2.imwrite("../results/equalized_normalized"+str(n)+"_yiq_"+image_file[23:], equalized_normalized_yiq[:, :, 0])

        equalized_normalized_hsv = normalization(equalized_hsv.copy())
        cv2.imwrite("../results/equalized_normalized"+str(n)+"_hsv_"+image_file[23:], equalized_normalized_hsv)
    '''

    # divisão da imagem em plaquinhas menores
    divided_rgb = division("../results/perspective_PCB_01_ilumin_03.jpg")
    # cv2.imwrite("../results/divided_rgb_"+image_file[23:], divided_rgb)
    cv2.namedWindow("divided", cv2.WINDOW_NORMAL)
    cv2.imshow("divided", divided_rgb)
    
    '''
    # thresholding
    images_file = ["../results/normalized3_equalized_bw_PCB_01_ilumin_06.jpg", "../results/normalized3_equalized_yiq_PCB_01_ilumin_06.jpg"]
    #print(images_file)
    for image_file in images_file:
        gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        th_mean = thresh(gray)
        #cv2.imwrite("../results/th_bin_"+image_file[11:], th_bin)
        #cv2.imwrite("../results/th_gaussian_"+image_file[11:], th_gaussian)
        cv2.imwrite("../results/th_mean_"+image_file[11:], th_mean)
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0

main()