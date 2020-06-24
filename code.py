import sys
import math
import cv2 as cv
import numpy as np
#import skimage.color as skc

# Carregando imagem de arquivo
def loadImage(fileName):
    srcGrey = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    srcRGB = cv.imread(fileName)
    # Colocar mais um parametro nesta função para dar opções de filtrar
    # srcGrey = cv.medianBlur(srcGrey, 11)
    # srcGrey = cv.GaussianBlur(srcGrey, (5, 5), 0)
    return srcGrey, srcRGB

def main(argv):
    print("Processamento de imagens de PCB.")
    print("Processado com:")
    print("Python:", sys.version)
    print("OprnCV: ", cv.__version__)
    print("NumPy: ", np.__version__)
    defaultFile = "bancoImgs/image_01.jpg"
    fileName = argv[0] if len(argv) > 0 else defaultFile
    # Load image
    srcGrey, srcRGB = loadImage(fileName)
    cv.namedWindow("Original Grey", cv.WINDOW_NORMAL)
    cv.imshow("Original Grey", srcGrey)
    cv.namedWindow("Original RGB", cv.WINDOW_NORMAL)
    cv.imshow("Original RGB", srcRGB)
    # Edges Detection
    #edges = edgesDetection(srcGrey)
    # thr1=50, thr2=100, apSize=3, L2Grad=True
    edges = cv.Canny(srcGrey, 50, 100, None, 3, True)
    cv.namedWindow("Canny Output", cv.WINDOW_NORMAL)
    cv.imshow("Canny Output", edges)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])