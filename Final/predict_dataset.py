import numpy as np
import os
import cv2
import click
from tqdm import tqdm
import tensorflow as tf

from skimage import exposure
from skimage.morphology import watershed
from scipy.ndimage.morphology import binary_fill_holes

from models import create_model, create_model_bf

BATCH_SIZE = 8
MARKER_THRESHOLD = 240


# [marker_threshold, cell_mask_threshold]
THRESHOLDS = {'DIC-C2DH-HeLa': [240, 216]}

def median_normalization(image):
    image_ = image / 255 + (.5 - np.median(image / 255))
    return np.maximum(np.minimum(image_, 1.), .0)


def hist_equalization(image):
    return cv2.equalizeHist(image) / 255


def get_normal_fce(normalization):
    if normalization == 'HE':
        return hist_equalization 
    if normalization == 'MEDIAN':
        return median_normalization
    else:
        return None


def remove_uneven_illumination(img, blur_kernel_size=501):
    """
    uses LPF to remove uneven illumination
    """
   
    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)
    
    img_blur = cv2.GaussianBlur(img_f, (blur_kernel_size, blur_kernel_size), 0)
    result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.int32)
    
    return result


def remove_edge_cells(label_img, border=20):
    edge_indexes = get_edge_indexes(label_img, border=border)
    return remove_indexed_cells(label_img, edge_indexes)


def get_edge_indexes(label_img, border=20):
    mask = np.ones(label_img.shape) 
    mi, ni = mask.shape
    mask[border:mi-border,border:ni-border] = 0
    border_cells = mask * label_img
    indexes = (np.unique(border_cells))

    result = []

    # get only cells with center inside the mask
    for index in indexes:
        cell_size = sum(sum(label_img == index))
        gap_size = sum(sum(border_cells == index))
        if cell_size * 0.5 < gap_size:
            result.append(index)
    
    return result


def remove_indexed_cells(label_img, indexes):
    mask = np.ones(label_img.shape)
    for i in indexes:
        mask -= (label_img == i)
    return label_img * mask


def get_image_size(path):
    """returns size of the given image"""

    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    return o.shape[0:2]


def get_new_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)


def read_image(path):
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img


def load_images(path, cut=False, new_mi=0, new_ni=0, normalization='HE', uneven_illumination=False):

    names = os.listdir(path)
    names.sort()

    mi, ni = get_image_size(path)

    dm = (mi % 16) // 2
    mi16 = mi - mi % 16
    dn = (ni % 16) // 2
    ni16 = ni - ni % 16

    total = len(names)
    normalization_fce = get_normal_fce(normalization)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)

    for i, name in enumerate(names):

        o = read_image(os.path.join(path, name))

        if o is None:
            print('image {} was not loaded'.format(name))

        if uneven_illumination:
            o = np.minimum(o, 255).astype(np.uint8)
            o = remove_uneven_illumination(o) 

        image_ = normalization_fce(o)

        image_ = image_.reshape((1, mi, ni, 1)) - .5
        image[i, :, :, :] = image_

    if cut:
        image = image[:, dm:mi16+dm, dn:ni16+dn, :]
    if new_ni > 0 and new_ni > 0:
        image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
        image2[:, :mi, :ni, :] = image
        image = image2

    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image

def postprocess_markers(img,
                        threshold=240,
                        erosion_size=12,
                        circular=True,
                        step=4):
    """
    erosion_size == c
    step == h
    threshold == tm
    """

    c = erosion_size
    h = step

    # original matlab code:
    # res = opening(img, size); % size filtering
    # res = hconvex(res, h) == h; % local contrast filtering
    # res = res & (img >= t); % absolute intensity filtering

    if circular:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c))
        markers = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        new_m = (hconvex(markers, h) == h).astype(np.uint8)
        glob_f = ((markers > threshold).astype(np.uint8) * new_m)

    # label connected components
    idx, markers = cv2.connectedComponents(glob_f)

    # print(threshold, c, circular, h)
    return idx, markers


# postprocess markers
def postprocess_markers2(img, threshold=240, erosion_size=12, circular=False, step=4):

    # distance transform | only for circular objects
    if circular:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        markers = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        new_m = ((hconvex(markers, step) > 0) & (img > threshold)).astype(np.uint8)
    else:
    
        # threshold
        m = img.astype(np.uint8)
        _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

        # filling gaps
        hol = binary_fill_holes(new_m*255).astype(np.uint8)

        # morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        new_m = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    idx, res = cv2.connectedComponents(new_m)

    return idx, res


def hmax2(img, h=50):

    h_img = img.astype(np.uint16) + h
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    rec0 = img

    # reconstruction
    for i in range(255):
        
        rec1 = np.minimum(cv2.dilate(rec0, kernel), h_img)
        if np.sum(rec0 - rec1) == 0:
            break
        rec0 = rec1
       
    # retype to uint8
    hmax_result = np.maximum(np.minimum((rec1 - h), 255), 0).astype(np.uint8)

    return hmax_result


def hconvex(img, h=5):
    return img - hmax2(img, h)

    
def hmax(ml, step=50):
    """
    old version of H-MAX transform
    not really correct
    """
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ml = cv2.blur(ml, (3, 3))
    
    rec1 = np.maximum(ml.astype(np.int32) - step, 0).astype(np.uint8)

    for i in range(255):
        rec0 = rec1
        rec1 = np.minimum(cv2.dilate(rec0, kernel), ml.astype(np.uint8))
        if np.sum(rec0 - rec1) == 0:
            break

    return ml - rec1 > 0 


def postprocess_cell_mask(b, threshold=230):

    # tresholding
    b = b.astype(np.uint8)
    bt = cv2.inRange(b, threshold, 255)

    return bt


def threshold_and_store(predictions,
                        input_images,
                        res_path,
                        thr_markers=240,
                        thr_cell_mask=230,
                        viz=False,
                        circular=False,
                        erosion_size=12,
                        step=4,
                        border=0):

    viz_path = res_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        m = predictions[i, :, :, 1] * 255
        c = predictions[i, :, :, 3] * 255


        # postprocess the result of prediction
        idx, markers = postprocess_markers2(m,
                                            threshold=thr_markers,
                                            erosion_size=erosion_size,
                                            circular=circular,
                                            step=step)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, markers)

        labels = watershed(-c, markers, mask=cell_mask)
        labels = remove_edge_cells(labels, border)


        # store result
        cv2.imwrite('{}/t{:03d}mask.tif'.format(res_path, i), labels.astype(np.uint8))
        
        viz_m = np.absolute(m - (markers > 0) * 64)

        if viz:
            o = (input_images[i, :, :, 0] + .5) * 255
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)

            labels_rgb = cv2.applyColorMap(labels.astype(np.uint8)*15, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

            m_rgb = cv2.cvtColor(viz_m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, c_rgb, overlay), 1)

            cv2.imwrite('{}/t{:03d}overlay.tif'.format(viz_path, i), overlay)
            cv2.imwrite('{}/t{:03d}o_rgb.tif'.format(viz_path, i), o_rgb)
            cv2.imwrite('{}/res_{:03d}.tif'.format(viz_path, i), result)
            cv2.imwrite('{}/markers_{:03d}.tif'.format(viz_path, i), markers.astype(np.uint8) * 16)


def predict_dataset(sequence, viz=False):
    """
    reads images from the path and converts them to the np array
    """
    sequence = str(sequence)

    dataset_name = 'DIC-C2DH-HeLa'

    erosion_size = 8
    NORMALIZATION = 'HE'
    MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['DIC-C2DH-HeLa']
    UNEVEN_ILLUMINATION = False
    CIRCULAR = False
    STEP = 0
    BORDER = 15
    process = threshold_and_store

    # load model
    model_path = os.path.join('./UNet_models_saved', 'UNet_DIC-C2DH-HeLa.h5')

    store_path = os.path.join('../images/', dataset_name, 'Sequence '+sequence+' Masks')
    
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    img_path = os.path.join('../images/', dataset_name, 'Sequence '+sequence)

    if not os.path.isdir(img_path):
        print('given name of dataset or the sequence is not valid')
        exit()

    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    input_img = load_images(img_path,
                            new_mi=new_mi,
                            new_ni=new_ni,
                            normalization=NORMALIZATION,
                            uneven_illumination=UNEVEN_ILLUMINATION)

    model = create_model(model_path, new_mi, new_ni)

    pred_img = model.predict(input_img, batch_size=BATCH_SIZE)
    print('pred shape: {}'.format(pred_img.shape))

    org_img = load_images(img_path)
    pred_img = pred_img[:, :mi, :ni, :]

    process(pred_img,
            org_img,
            store_path,
            thr_markers=MARKER_THRESHOLD,
            thr_cell_mask=C_MASK_THRESHOLD,
            viz=viz,
            circular=CIRCULAR,
            erosion_size=erosion_size,
            step=STEP,
            border=BORDER)
