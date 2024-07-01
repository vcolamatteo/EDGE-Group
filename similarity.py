import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2

model = models.resnext50_32x4d(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model.eval()

# Define a transform to preprocess the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def convert_to_PIL(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def extract_features(frame):
    #image = Image.open(image_path)
    image = preprocess(convert_to_PIL(frame)).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    features = features.squeeze().numpy()  # Remove batch dimension and convert to numpy
    
    return features


def similarity(feature_init, frame):
    # Example image paths

    # Extract features for all images
    feature_frame = extract_features(frame)

    # Compute cosine similarities between the feature vectors
    similarities = cosine_similarity([feature_init],[feature_frame])[0][0]
    
    return similarities