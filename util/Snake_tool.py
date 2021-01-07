import numpy as np
import torch
"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F"""
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score as normalized_mutual_info_score
from scipy import stats
grid_sample = False
gaussian = False
dot = False
crop = False
rotate = False
SLIC = False
get_var = False
MI = False
P2C = True

"""def get_grid(w, h, x, y, crop_w, crop_h, grid_w, grid_h):
    ax = 1 / (w/2)
    bx = -1
    ay = 1 / (h/2)
    by = -1
    left_x = x - (crop_w/2)
    right_x = x + (crop_w/2)
    left_y = y - (crop_h/2)
    right_y = y + (crop_h/2)
    left_x = left_x*ax + bx
    right_x = right_x*ax + bx
    left_y = left_y*ay + by
    right_y = right_y*ay + by
    grid_x = torch.linspace(float(left_x), float(right_x), grid_w)
    grid_y = torch.linspace(float(left_y), float(right_y), grid_h)
    meshy, meshx = torch.meshgrid((grid_y, grid_x))
    grid = torch.stack((meshx, meshy), 2)
    grid = grid.unsqueeze(0)  # add batch dim
    return grid"""


def get_gaussian(x, y, var, kernel_size, image_size):

    img_gaus = np.zeros((image_size, image_size))

    for i in range(image_size):
        for j in range(image_size):
            # if(((i-x)**2+(j-y)**2) < kernel_size**2):
            img_gaus[j][i] = np.exp(-(((i-x)**2+(j-y)**2)/(2*(var**2))))

    img_gaus = np.expand_dims(img_gaus, 2)

    return img_gaus


def orientation_dot(v1, v2):
    normalize_action1 = v1/np.linalg.norm(v1)
    normalize_target1 = v2/np.linalg.norm(v2)
    dot_orien1 = np.dot(normalize_action1, normalize_target1)
    return dot_orien1


def outofbound(x, y):

    if x < 0 or x > 127 or y < 0 or y > 127:
        return True
    else:
        return False


def P2C(r, theta, center):
    x = r*np.cos(theta) + center[0]
    y = r*np.sin(theta) + center[1]
    if(x > center[0]):
        x -= 1
    if(y > center[1]):
        y -= 1
    return int(x), int(y)


def cropped(img, x, y, cropped_w, w):
    # x, y = P2C(r, theta, center)
    cropped_img = np.zeros((cropped_w, cropped_w, 3), np.float32)
    l_x = x - int(cropped_w/2)
    r_x = x + int(cropped_w/2)
    t_y = y - int(cropped_w/2)
    b_y = y + int(cropped_w/2)

    # x to cropped_x
    if(l_x < 0):
        cropped_l_x = np.abs(l_x)
        l_x = 0
    elif(0 <= l_x < 128):
        cropped_l_x = 0
    else:
        cropped_l_x = -1
    if(r_x > w):
        cropped_r_x = cropped_w - (np.abs(r_x) - w)
        r_x = w
    elif(r_x < 0):
        cropped_r_x = -1
    else:
        cropped_r_x = cropped_w
    if(t_y < 0):
        cropped_t_y = np.abs(t_y)
        t_y = 0
    elif(0 <= t_y < 128):
        cropped_t_y = 0
    else:
        cropped_t_y = -1
    if(b_y > w):
        cropped_b_y = cropped_w - (np.abs(b_y) - w)
        b_y = w
    elif(b_y < 0):
        cropped_b_y = -1
    else:
        cropped_b_y = cropped_w
    # print(img.shape)

    if(cropped_t_y == -1 or cropped_l_x == -1 or cropped_b_y == -1 or cropped_r_x == -1):
        return cropped_img
    else:
        cropped_img[cropped_t_y:cropped_b_y, cropped_l_x:
                    cropped_r_x] = img[t_y:b_y, l_x:r_x].copy()
        return cropped_img


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def Super_pixel(image, layer):
    segments_slic = slic(image, n_segments=250, compactness=10, sigma=1)
    s_img = color.label2rgb(segments_slic, image, kind='avg')
    if layer == 0:
        return s_img
    segments_slic_1 = slic(s_img, n_segments=100, compactness=10, sigma=1)
    s_img_1 = color.label2rgb(segments_slic_1, s_img, kind='avg')
    if layer == 1:
        return s_img_1
    segments_slic_2 = slic(s_img_1, n_segments=50, compactness=10, sigma=1)
    s_img_2 = color.label2rgb(segments_slic_2, s_img_1, kind='avg')
    if layer == 2:
        return s_img_2
    segments_slic_3 = slic(s_img_2, n_segments=10, compactness=10, sigma=1)
    s_img_3 = color.label2rgb(segments_slic_3, s_img_2, kind='avg')
    if layer == 3:
        return s_img_3
    segments_slic_4 = slic(s_img_3, n_segments=5, compactness=10, sigma=1)
    s_img_4 = color.label2rgb(segments_slic_4, s_img_3, kind='avg')
    if layer == 4:
        return s_img_4


def get_color_var(image, points):
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[np.where(mask > 0)] = image[np.where(mask > 0)]
    mask_size = mask[np.where(mask > 0)].size
    var = np.var(image[np.where(mask > 0)], ddof=0)
    # divide size to average
    return np.sqrt(var), mask_size

def get_featmap_8(model, image):
    vgg_model = model.cuda()
    tensor_img = transforms.ToTensor()(image).unsqueeze_(0).cuda()
    feature_map = vgg_model(tensor_img) # (1,512,8,8)# 3,8,15,22,29
    return feature_map 

def bilitlp(feature_map, coord, img_width):
    x, y = coord[0], coord[1]
    if(outofbound(x, y)):
        print(coord)
    _, _, H, W = feature_map.shape
    ratio = img_width/H
    feature_coord_y = int(y/ratio)
    feature_coord_x = int(x/ratio)
    w_y = int(y % ratio)
    w_x = int(x % ratio)

    if(feature_coord_y - 1 < 0 or feature_coord_x - 1 < 0):
        left_top = torch.zeros(feature_map[:, :, 0, 0].shape).cuda()
    else:
        left_top = feature_map[:, :, feature_coord_y - 1, feature_coord_x - 1]
    if(feature_coord_y < H or feature_coord_x - 1 > 0):
        left_bot = feature_map[:, :, feature_coord_y, feature_coord_x - 1]
    else:
        left_bot = torch.zeros(left_top.shape).cuda()
    if(feature_coord_x < H or feature_coord_y - 1 > 0):
        rght_top = feature_map[:, :, feature_coord_y - 1, feature_coord_x]
    else:
        rght_top = torch.zeros(left_top.shape).cuda()
    if(feature_coord_y < H and feature_coord_x < H):
        rght_bot = feature_map[:, :, feature_coord_y, feature_coord_x]
    else:
        rght_bot = torch.zeros(left_top.shape).cuda()
    if(w_y != 0):
        left_itpl = left_top * (ratio-w_y)/ratio + left_bot*(w_y)/ratio
        rght_itpl = rght_top * (ratio-w_y)/ratio + rght_bot*(w_y)/ratio
    else:
        left_itpl = left_bot
        rght_itpl = rght_bot
    if(w_x != 0):
        bi_itpl = left_itpl * (ratio-w_x)/ratio + rght_itpl*(w_x)/ratio
    else:
        bi_itpl = rght_itpl

    return bi_itpl

def mi(x, y):
    #c_xy = np.histogram2d(x, y, bins)[0]
    return normalized_mutual_info_score(x,y)


if __name__ == "__main__":
    path = "C:\\Users\\Mislab\\Desktop\\Research\\dataset\\LFW\\test\\image/Ahmed_Chalabi_0005.jpg"
    # grid create function test
    if(grid_sample):
        img = cv2.imread(path)
        img = np.asarray(img).astype(float)
        img = torch.FloatTensor(np.expand_dims(img, 0))
        img = img.permute(0, 3, 1, 2)

        grid1 = get_grid(128, 128, 25, 25, 32, 32, 32, 32)

        cropped1 = F.grid_sample(img, grid1)
        cropped1 = torch.squeeze(cropped1)
        cropped1 = cropped1.permute(1, 2, 0)

        # print(cropped1)
        img = img.permute(0, 2, 3, 1)
        img = torch.squeeze(img)
        print(img.shape)
        img = cropped1.cpu().detach().numpy()
        cv2.imshow("image", img)
        cv2.waitKey()
    # gaussian create function test
    if(gaussian):
        img = get_gaussian(110, 64, 16, 32, 128)
        img = np.asarray(img).astype(float)
        print(img.shape)
        cv2.imshow("image", img)
        cv2.waitKey()

    if(dot):
        v1 = [2, 3]
        v2 = [-1, 1]
        print(orientation_dot(v1, v2))
    if(crop):
        img = cv2.imread(path)
        crop_img = cropped(img, 117, 41, 16, 128)
        cv2.imshow("image", crop_img)
        cv2.waitKey()
    if(rotate):
        img = cv2.imread(path)
        rotated_image = rotate_image(img, 0)
        cv2.imshow("image", rotated_image)
        cv2.waitKey()
        rotated_image = rotate_image(img, 45)
        cv2.imshow("image", rotated_image)
        cv2.waitKey()
    if(SLIC):
        img = cv2.imread(path)
        #img = np.asarray(img).astype(float)
        s_img = Super_pixel(img, 2)
        cv2.imshow("image", s_img)
        cv2.waitKey()
    if(get_var):
        img = cv2.imread(path)
        # points = [[64, 64], [112, 64], [88, 88]]
        points = [[64, 64], [64, 80], [80, 64]]
        print(points[0][0])
        x = [points[0][0], points[1][0], points[2][0]]
        y = [points[0][1], points[1][1], points[2][1]]
        max_x, min_x = max(x), min(x)
        max_y, min_y = max(y), min(y)
        img_in = img[min_x:max_x, min_y:max_y]
        points = np.array(points, np.int32)
        new_points = np.subtract(points, [min_x, min_y])
        print(new_points)
        var = get_color_var(img_in, new_points)
        var = get_color_var(img, points)
        print(var)
    if(MI):
        a = np.array([0.1,0.2,0.3])
        b = np.array([0.7,0.7,0.1])
        #print(stats.entropy([0.5,0.5])) # entropy of 0.69, expressed in nats
        #print(mutual_info_classif(a.reshape(-1,1), b, discrete_features = True)) # mutual information of 0.69, expressed in nats
        print(mutual_info_score(a,b))