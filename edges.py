import cv2 as cv
import numpy as np
import math
import sys
import imutils

def load_image(image_file):
    # ler imagem
    image = cv.imread(image_file, cv.IMREAD_COLOR)
    return image

def thresholding(image):
    # transforma imagem rgb em hsv
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # binarização da imagem
    green_lower = (60, 60, 20)
    green_upper = (100, 255, 140)
    mask = cv.inRange(hsv, green_lower, green_upper)

    # contornos
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # verificar areas dos contornos e cortar
    if len(contours) > 0:
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if area > 800000:
                #print(area)
                x, y, w, h = cv.boundingRect(contours[i])
                cut_image = image[y-40:y+h+40, x-40:x+w+40, :]
                # desenhar contorno da placa
                contour = cv.drawContours(image.copy(), contours, i, (255,255,255), 3)
                contour = contour[y-40:y+h+40, x-40:x+w+40, :]

    return contour, cut_image

def hough(image):
    # Edges Detection
    # thr1=50, thr2=100, apSize=3, L2Grad=True
    edges = cv.Canny(image, 50, 100, None, 3, True)
    lines = cv.HoughLines(edges, 1, np.pi/180, 400)

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # descobrir calculo que acontece aqui pra consertar
            pt1 = (int(x0 + 4000*(-b)), int(y0 + 4000*(a)))
            pt2 = (int(x0 - 4000*(-b)), int(y0 - 4000*(a)))
            cv.line(image, pt1, pt2, (255,255,255), 3)

    return image

def probabilistic_hough(image):
    # Edges Detection
    # thr1=50, thr2=100, apSize=3, L2Grad=True
    edges = cv.Canny(image, 50, 100, None, 3, True)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=400, maxLineGap=50)

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(image, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3)
    return image

def detect_lines(image_file="../imagens/image_01.jpg"):
    print("Processamento de imagens de PCB.")
    print("Processado com:")
    print("Python:", sys.version)
    print("OpenCV: ", cv.__version__)
    print("NumPy: ", np.__version__)

    # carregar imagem
    image = load_image(image_file)
    print("Imagem: ", image.shape)

    # cortar
    contour, cut_image = thresholding(image.copy())
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.imshow("image", contour)
    cv.namedWindow("cut_image", cv.WINDOW_NORMAL)
    cv.imshow("cut_image", cut_image)

    # transformar imagem cortada para b&w
    cut_gray = cv.cvtColor(cut_image, cv.COLOR_BGR2GRAY)
    print("Imagem cortada: ", cut_gray.shape)

    # zoom sobre imagem cortada b&w
    zoom = (int(cut_gray.shape[1]*1.5), int(cut_gray.shape[0]*1.5))
    gray = cv.resize(cut_gray, zoom, interpolation=cv.INTER_NEAREST)

    # detectar linhas (hough transform)
    lines = hough(gray.copy())
    cv.namedWindow("hough", cv.WINDOW_NORMAL)
    cv.imshow("hough", lines)

    # detectar linhas (hough probabilistic transform)
    lines_p = probabilistic_hough(gray.copy())
    cv.namedWindow("probabilistic_hough", cv.WINDOW_NORMAL)
    cv.imshow("probabilistic_hough", lines_p)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0

detect_lines('../imagens/PCB_01_ilumin_06.jpg')