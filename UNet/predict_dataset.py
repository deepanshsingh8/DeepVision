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
from deepwater.src.evaluation import threshold_and_store as ts_Fluo

BATCH_SIZE = 8
MARKER_THRESHOLD = 240


# [marker_threshold, cell_mask_threshold]
THRESHOLDS = {
    'BF-C2DL-MuSC': [245, 252],
    'BF-C2DL-HSC': [200, 254],
    'DIC-C2DH-HeLa': [240, 216],
    'Fluo-N2DH-SIM+': [10, 200],
    'PhC-C2DL-PSC': [240, 156]}

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
    if 'Fluo' in path:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if 'Fluo-N2DH-SIM+' in path:
            img = np.minimum(img, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img


# read images
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

    print(predictions.shape)
    print(input_images.shape)
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
        cv2.imwrite('{}/mask_{:03d}.tif'.format(res_path, i), labels.astype(np.uint16))
        
        viz_m = np.absolute(m - (markers > 0) * 64)

        if viz:
            o = (input_images[i, :, :, 0] + .5) * 255
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)

            labels_rgb = cv2.applyColorMap(labels.astype(np.uint8)*15, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

            m_rgb = cv2.cvtColor(viz_m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, c_rgb, overlay), 1)
            cv2.imwrite('{}/res_{:03d}.tif'.format(viz_path, i), result)
            cv2.imwrite('{}/markers_{:03d}.tif'.format(viz_path, i), markers.astype(np.uint8) * 16)


def threshold_and_store_bf(predictions,
                        original=None,
                        out_path='.',
                        thr_markers=240,
                        thr_cell_mask=230,
                        viz=False,
                        circular=False,
                        erosion_size=12,
                        step=4,
                        merge_method='watershed',
                        start_index=0,
                        n_digits=3):

    viz_path = out_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        if predictions.shape[3] > 2:
            m = (predictions[i, :, :, 1] * 255).astype(np.uint8)
            c = (predictions[i, :, :, 3] * 255).astype(np.uint8)
        else:
            m = predictions[i, :, :, 0]
            c = predictions[i, :, :, 1]

        # postprocess the result of prediction
        idx, marker_function = postprocess_markers(m,
                                                   threshold=thr_markers,
                                                   erosion_size=erosion_size,
                                                   circular=circular,
                                                   step=step)

        # print(np.unique(c), thr_cell_mask)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, marker_function)

        # segmentation function
        # clipping
        segmentation_function = np.maximum((255 - c), (255 - cell_mask))

        assert merge_method in ['watershed', 'topdist', 'eucldist'], merge_method

        if merge_method == 'watershed':
            labels = watershed(segmentation_function, marker_function, mask=cell_mask)
        elif merge_method == 'topdist':
            labels = topdist_merge(marker_function, cell_mask)
        elif merge_method == 'eucldist':
            labels = eucldist_merge(marker_function, cell_mask)
        else:
            labels = None


        # store result
        cv2.imwrite('{}/mask_{:04d}.tif'.format(out_path, i + start_index), labels.astype(np.uint16))

        if viz:
            # vizualize result in rgb
            o = (original[i, :, :, 0] + .5) * 255
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)
            m_rgb = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)

            fg_mask = (labels != 0).astype(np.uint8)
            bg_mask = (labels == 0).astype(np.uint8)

            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] * fg_mask
            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] + bg_mask * 180
            labels_rgb[:, :, 1] = labels_rgb[:, :, 1] + bg_mask * 180
            labels_rgb[:, :, 2] = labels_rgb[:, :, 2] + bg_mask * 180

            overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

            mf_rgb = cv2.cvtColor(((marker_function > 0) * 255 ).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            sf_rgb = cv2.cvtColor(segmentation_function.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, mf_rgb, c_rgb, sf_rgb, overlay), 1)
            cv2.imwrite('{}/res_{:04d}.tif'.format(viz_path, i + start_index), result)
            cv2.imwrite('{}/c_{:04d}.tif'.format(viz_path, i + start_index), c.astype(np.uint8))
            cv2.imwrite('{}/m_{:04d}.tif'.format(viz_path, i + start_index), m.astype(np.uint8))

            cv2.imwrite('{}/marker_fc_{:04d}.tif'.format(viz_path, i + start_index), marker_function)
            cv2.imwrite('{}/segmen_fc_{:04d}.tif'.format(viz_path, i + start_index), segmentation_function)
            cv2.imwrite('{}/cell_mask_{:04d}.tif'.format(viz_path, i + start_index), cell_mask)


@click.command()
@click.option('--name', prompt='path to dataset',
              help='Path to the dataset.')
@click.option('--sequence', prompt='sequence number',
              help='Dataset number.')
@click.option('--viz/--no-viz', default=True,
              help='true if produces also viz images.')
def predict_dataset(name, sequence, viz=True):
    """
    reads images from the path and converts them to the np array
    """

    dataset_path = name

    # check if there is a model for this dataset
    if not os.path.isdir(dataset_path):
        print('there is no solution for this dataset')
        exit()

    erosion_size = 1
    if 'DIC-C2DH-HeLa' in name:
        erosion_size = 8
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['DIC-C2DH-HeLa']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = False
        STEP = 0
        BORDER = 15
        MODEL = 4
        process = threshold_and_store
                
    elif 'Fluo-N2DH-SIM+' in name:
        erosion_size = 5
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['Fluo-N2DH-SIM+']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = True
        STEP = 1
        BORDER = 1
        MODEL = 4
        process = ts_Fluo

    elif 'PhC-C2DL-PSC' in name:
        erosion_size = 1
        NORMALIZATION = 'MEDIAN'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['PhC-C2DL-PSC']
        UNEVEN_ILLUMINATION = True
        CIRCULAR = True
        STEP = 3
        BORDER = 12
        MODEL = 4
        process = threshold_and_store

    elif 'BF-C2DL-HSC' in name:
        erosion_size = 10
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['BF-C2DL-HSC']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = True
        STEP = 1
        BORDER = 1
        MODEL = 3

    elif 'BF-C2DL-MuSC' in name:
        erosion_size = 9
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD = THRESHOLDS['BF-C2DL-MuSC']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = True
        STEP = 15
        BORDER = 1
        MODEL = 3
    else:
        print('unknown dataset')
        return

    # load model
    if MODEL == 4:
        model_name = [name for name in os.listdir(dataset_path) if 'PREDICT' in name]
        assert len(model_name) == 1, 'ambiguous choice of nn model, use keyword "PREDICT" exactly for one ".h5" file'
        model_init_path = os.path.join(dataset_path, model_name[0])

    if MODEL == 3:
        seg_model_name = [name for name in os.listdir(dataset_path) if 'SEG_PREDICT' in name]
        det_model_name = [name for name in os.listdir(dataset_path) if 'DET_PREDICT' in name]
        assert len(seg_model_name) == 1, 'ambiguous choice of seg model, use keyword "SEG_PREDICT"' \
                                         ' exactly for one ".h5" file'
        assert len(det_model_name) == 1, 'ambiguous choice of det model, use keyword "DET_PREDICT"' \
                                         ' exactly for one ".h5" file'

    store_path = os.path.join('..', name, 'MU-Lux-CZ', '{}_RES'.format(sequence))
    
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    img_path = os.path.join('..', name, 'MU-Lux-CZ', sequence)
    if not os.path.isdir(img_path):
        print('given name of dataset or the sequence is not valid')
        exit()

    # if not os.path.isfile(os.path.join(name, 'model_init.h5')):
    #     print('directory {} do not contain model file'.format(name))

    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    print(mi, ni)
    print(new_mi, new_ni)

    input_img = load_images(img_path,
                            new_mi=new_mi,
                            new_ni=new_ni,
                            normalization=NORMALIZATION,
                            uneven_illumination=UNEVEN_ILLUMINATION)

    if MODEL == 3:

        seg_path = os.path.join(dataset_path, seg_model_name[0])
        det_path = os.path.join(dataset_path, det_model_name[0])

        model_seg = create_model_bf(seg_path, new_mi, new_ni)
        model_det = create_model_bf(det_path, new_mi, new_ni)

        batch_size = 40

        img_number = len(input_img)

        for bindex in range(0, img_number, batch_size):
            images = input_img[bindex:(bindex + batch_size)]
            print(f'{bindex} / {img_number}')

            pred_seg = model_seg.predict(images, batch_size=4)
            pred_seg = pred_seg[:, :, :, 2:]
            print('seg predicted')
            pred_det = model_det.predict(images, batch_size=4)
            pred_det = pred_det[:, :, :, 2:]
            print('det predicted')
            pred_img = np.concatenate((pred_det, pred_seg), axis=3)

            print('pred shape: {}'.format(pred_img.shape))

            pred_img = pred_img[:, :mi, :ni, :]
            print(pred_img.shape)
            images = images[:, :mi, :ni, :]
            print(images.shape)

            threshold_and_store_bf(pred_img,
                                   images,
                                   store_path,
                                   thr_markers=MARKER_THRESHOLD,
                                   thr_cell_mask=C_MASK_THRESHOLD,
                                   viz=viz,
                                   circular=CIRCULAR,
                                   erosion_size=erosion_size,
                                   step=STEP,
                                   start_index=bindex,
                                   n_digits=4)

    if MODEL == 4:
        model = create_model(model_init_path, new_mi, new_ni)

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


if __name__ == "__main__":
    predict_dataset()
