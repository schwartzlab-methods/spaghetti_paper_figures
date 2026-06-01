'''
Various cell segmentation methods
'''
import numpy as np
import torch
import cellpose.models
import squidpy as sq
from pycocotools.coco import COCO
from stardist.models import StarDist2D
from csbdeep.utils import normalize

def compute_dice_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return (2 * intersection) / (mask1.sum() + mask2.sum())

def compute_probabilistic_dice_score(pred, gt, epsilon=1e-6):
    """
    Compute the Probabilistic Dice Score between a soft prediction and ground truth.
    
    :param pred: numpy array, predicted probabilities (values between 0 and 1)
    :param gt: numpy array, ground truth probabilities (values between 0 and 1)
    :param epsilon: small constant to avoid division by zero
    :return: Probabilistic Dice Score
    """
    numerator = 2 * np.sum(pred * gt)
    denominator = np.sum(pred ** 2) + np.sum(gt ** 2) + epsilon
    return numerator / denominator

def get_mask_per_instance(coco_L: list[COCO], f_path: str):
    '''
    Get the masks of the image in the COCO format
    '''
    f_name = f_path.split("/")[-1]
    if f_name.endswith('.png'):
        f_name = f_name.replace('.png', '.tif')
    # get the image id
    image_id = None
    for coco in coco_L:
        for each in coco.imgs.values():
            if each["file_name"] == f_name:
                image_id = each["id"]
                current_coco = coco
                break
        if image_id is not None:
            break
    if image_id is None:
        raise ValueError("Image not found in the COCO dataset")
    # get all segmentations of the objects in the image
    ann_ids = current_coco.getAnnIds(imgIds=image_id)
    anns = current_coco.loadAnns(ann_ids)
    mask = current_coco.annToMask(anns[0]) # shape: (256,256,3)
    for ann in anns[1:]:
        mask += current_coco.annToMask(ann)
    return mask

def build_segment_model(name, model_type="stardist"):
    if model_type == "stardist":
        return StarDist2D.from_pretrained(name)
    elif model_type == "cellpose":
        return cellpose.models.CellposeModel(model_type=name, gpu=torch.cuda.is_available())
    else:
        raise ValueError(f"Model type ({model_type}) not supported")

def stardist_he(img, model) -> np.ndarray:
    '''
    Run stardist on the image
    '''
    # model.thresholds = {'prob': 0.2, 'nms': 0.5}
    labels, _ = model.predict_instances(normalize(img, 1,99.8, axis=(0,1)),
                                        prob_thresh=0.01, nms_thresh=0.6)
    return labels

def cellpose_he(img, model, min_size=10, flow_threshold=4.0, 
                channel_cellpose=0, start=0, batch_size=1):
    res, _, _ = model.eval(
        img,
        batch_size=batch_size,
        channels=[channel_cellpose, 0],
        diameter=None,
        min_size=min_size,
        invert=True,
        flow_threshold=flow_threshold,
        cellprob_threshold=-5.8
    )
    # add start to every segment except the background (0)
    res = res + start
    res[res == start] = 0
    print(f"Number of segments: {len(np.unique(res))}")
    return res

def deepsea(img, model):
    # get original size
    img_size = img.shape[0]
    if np.max(img) < 1:
        img = (img * 255).astype(np.uint8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ensure 1 channel only
    if len(img.shape) == 3 and img.shape[2] > 1:
        img = np.mean(img, axis=2, keepdims=True).astype(np.uint8)
    img = (255 * ((img - img.min()) / (np.ptp(img) + 1e-8))).astype(np.uint8)
    test_transforms = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((383,512)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = [0.5],
                                                    std = [0.5])
                           ])
    tensor_img=test_transforms(img).to(device=device, dtype=torch.float32)
    mask_pred, edge_pred = model(tensor_img.unsqueeze(0))
    mask_pred = mask_pred.argmax(dim=1).cpu().numpy()
    label_img, _ = ndi.label(remove_small_objects(mask_pred[0, :, :] > 0, min_size=20, connectivity=1))
    # resize
    out = np.resize(label_img, (img_size, img_size))
    return out

def segment_watershed(image: np.ndarray, if_mic: bool) -> np.ndarray:
    '''
    Segment using Watershed method
    '''
    x = image.shape[0]
    y = image.shape[1]
    # load image
    img = sq.im.ImageContainer(image)
    # use channel_cellpose = 0 for grayscale segmentation
    if if_mic:
        sq.im.segment(img=img, layer="image", channel=None, method="watershed")
    else:
        # sq.im.segment(img=img, layer="image", channel=None, method=cellpose_he)
        sq.im.process(img, layer="image", method="smooth", sigma=4)
        sq.im.segment(img=img, layer="image_smooth", method="watershed", thresh=None, geq=False)
    # print(f"Number of segments: {len(np.unique(img['segmented_custom']))}")
    print(f"Number of segments: {len(np.unique(img['segmented_watershed']))}")
    # return the segmented image as a numpy array in (x, y) format
    # ret = np.array(img['segmented_custom']).reshape(x, y)
    ret = np.array(img['segmented_watershed']).reshape(x, y)
    return ret

def crop(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    '''
    crop the image that is in (X, Y, C) by removing the background (all channles black) 
    according to the masks provided in (X, Y) form
    '''
    # Find the indices of the non-zero elements
    nz = np.argwhere(masks)
    x_min, y_min = nz.min(axis=0)
    x_max, y_max = nz.max(axis=0)
    cropped = image[x_min-10:x_max+10, y_min-10:y_max+10]

    return np.array(cropped)
