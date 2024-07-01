from model_training.tracker.fear_tracker import FEARTracker
from hydra.utils import instantiate
from model_training.utils.torch import load_from_lighting
from model_training.utils.hydra import load_hydra_config_from_path
import cv2
import numpy as np
from typing import List

def get_tracker(config_path: str, config_name: str, weights_path: str) -> FEARTracker:
    config = load_hydra_config_from_path(config_path=config_path, config_name=config_name)
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path).cuda().eval()
    tracker: FEARTracker = instantiate(config["tracker"], model=model)

    return tracker


def track_frame(tracker, frame) -> List[np.ndarray]:
    
    t=tracker.update(frame)    
    tracked_bbox = t[0]["bbox"]
    score=t[1].detach().cpu().numpy()
    
    return tracked_bbox, score


def draw_bbox(image, bbox, width= 5, color= (0, 255, 0)) -> np.ndarray:

    x, y, w, h = bbox
    return cv2.rectangle(image, (x, y), (x + w, y + h), color, width)
