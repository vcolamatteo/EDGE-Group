import numpy as np
import cv2
from PIL import Image

from segment_anything import sam_model_registry, SamPredictor


def loadPredictor(sam_checkpoint="/home/vc/Documents/segment-anything/weights/sam_vit_b_01ec64.pth", model_type="vit_b", device="cuda"):

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    return mask_image

def show_box(img, box):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    
    cv2.rectangle(np.array(img), (x0, y0), (x0+w, y0+h), (255, 0, 0), 3)

def bounding_box(mask):
    true_indices = np.argwhere(mask)
    
    if len(true_indices) == 0:
        return (0, 0, 0, 0)
    
    (y, x) = true_indices.T
    x_min, y_min = np.min(x), np.min(y)
    x_max, y_max = np.max(x), np.max(y)
    
    return (x_min, y_min, x_max, y_max)

def segment(image, input_point, predictor):
    predictor.set_image(image)

    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = show_mask(mask)[:, :, 0]
        mask[mask != 0] = 255

        image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        overlay_color = [255, 0, 0, 128]
        overlay = np.zeros_like(image_rgba, dtype=np.uint8)
        overlay[:, :] = overlay_color

        expanded_mask = mask[:, :, np.newaxis]
        combined_rgba = np.where(expanded_mask, overlay, image_rgba)

        combined_image = Image.fromarray(combined_rgba)
        bbox = bounding_box(mask)
        #show_box(combined_image, bbox)

    return combined_image, [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]