import os
import cv2
import numpy as np
import glob

app_win_size_x = 870
app_win_size_y = 500
γ_range_upper_bound = 5

output_dir_path = output_pl_path = output_he_path = output_lo_path = ""

@staticmethod
def root_path(): #當前 working dir 之 root path
    return os.getcwd()
    #return "/workspaces/mvl/ImageProcessing"

def set_output_path():
    global output_dir_path, output_pl_path, output_he_path, output_lo_path
    output_dir_path = os.path.join(root_path(), "output")
    output_pl_path = os.path.join(output_dir_path, "power-law")
    output_he_path = os.path.join(output_dir_path, "histogram_equalization")
    output_lo_path = os.path.join(output_dir_path, "Laplacian_operator")
    
def get_output_path():
    global output_dir_path, output_pl_path, output_he_path, output_lo_path
    return [output_dir_path, output_pl_path, output_he_path, output_lo_path]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("mkdir "+dir_path)    
    else:
        print(dir_path+" already exist, no need to mkdir.")

def get_image_path(path): #root_path/HW1_test_image
    return glob.glob(os.path.join(path, "*.bmp"))

def show_img_fullscreen(img_name, showimg ,type):
    cv2.namedWindow(img_name, type)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(img_name, app_win_size_x,app_win_size_y)
    #cv2.moveWindow(img_name, app_pos_x,app_pos_y)
    cv2.imshow(img_name, showimg)

def read_and_operate_image(image_path):
    image =cv2.imread(image_path)
    #show_img_fullscreen("Current Image: "+image_path, image, cv2.WINDOW_KEEPRATIO)
    image_gray =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #show_img_fullscreen("Current Image(grayscale): "+image_path, image_gray, cv2.WINDOW_KEEPRATIO)
    return image, image_gray

#algorithm implementation
#power-law
"""-------------------------------
formula:
s = c(r+ε)^γ
----------------------------------
s: output
c: constant, default = 1
ε: offset, default = 0
γ: gamma, user-defined
c and γ are positive constants.
-------------------------------"""
def power_law_transform(img, γ):
    if isinstance(img, np.ndarray) and isinstance(γ, float):
        return np.array(255*(img/255)**γ, dtype='uint8')
    else:
        print("img: "+str(type(img)),end=" ")
        print("γ: "+str(type(γ)))
        print("type error.")
        exit()

#Histogram Equalization
"""-------------------------------
先把所有顏色的數量都統計成直方圖，並計算每個顏色的機率(像素個數/像素總數)。
再計算累計機率，乘上像素總數並4捨5入得到新的像素值。
-------------------------------"""
def histogram_equalization(img):
    hist, bins= np.histogram(img.ravel(), 256, [0, 255])    #直方圖 灰階:256bins，0 ~ 255
    pdf = hist/img.size   #計算每個顏色機率
    cdf = pdf.cumsum()
    equ_value = np.round((cdf*255)).astype('uint8')
    return equ_value[img]

#laplacian_operator
"""-------------------------------
padding先把矩陣補成(m+p*2)x(n+p*2)，convolution之後會變回mxn。
p = LM 的 kernel size/2
ex:
Laplacian matrix 3x3 (kernel size 3)
LM = [[0, 1, 0],[1, -4, 1], [0, 1, 0]]
p = 1
----------------------------------
formula:
new_point = c·sum(LM convolution img) + point

計算後的值可能會超出範圍、以上下值取代。
-------------------------------"""
def laplacian_operator(img):
    M, N = img.shape
    k=3 # kernel
    c = -1
    p = k // 2
    new_img = np.zeros((M + p * 2, N + p * 2), dtype=np.float64)
    new_img[p: p + M, p: p + N] = img.copy().astype(np.float64)
    tmp = new_img.copy()

    # laplacian matrix
    LM = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]
    for m in range(M):
        for n in range(N):
            point = tmp[p + m, p + n]
            con = tmp[m: m + k, n: n + k]
            new_img[p + m, p + n] = c*np.sum(LM*con) + point
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img[p: p + M, p: p + N].astype('uint8')

    return new_img

#參考: https://www.geeksforgeeks.org/python-intensity-transformation-operations-on-images/
#參考: https://codeinfo.space/imageprocessing/histogram-equalization/
#參考: https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/

def choose_pl(img, range_upper_bound):
    record = 999
    value = 0
    range_upper_bound *=10
    best = img
    for γ in range(0, range_upper_bound ,1):
        temp_img = power_law_transform(img, γ/10)
        cal = abs(np.mean(temp_img)-128)
        #print(γ/10, end=": ")
        #print(cal)
        if  cal < record:
            record = cal
            best = temp_img
            value = γ/10

    print("best record=",end="")
    print(record,end=",")
    print(" γ = ", end="")
    print(value)

    return best

# main
image_dir = os.path.join(root_path(), "HW1_test_image")
print(image_dir)
images = get_image_path(image_dir)  #取得圖片路徑
print(images)

set_output_path()
for output_path in get_output_path():
    mkdir(output_path)

dir, pl, he, lo = get_output_path()

for image in images:
    file = image.replace(".bmp", "")
    img, img_gray = read_and_operate_image(image)
    img_pl = choose_pl(img, γ_range_upper_bound)
    img_he = histogram_equalization(img)
    #img_heg = histogram_equalization(img_gray)
    img_lo = laplacian_operator(img_gray)

    #show_img_fullscreen("img_he", img_he, cv2.WINDOW_KEEPRATIO)

    cv2.imwrite(file.replace(image_dir, pl)+"_power_law_transform.bmp", img_pl)
    cv2.imwrite(file.replace(image_dir, he)+"_histogram_equalization.bmp", img_he)
    cv2.imwrite(file.replace(image_dir, lo)+"_laplacian_operator.bmp", img_lo)