import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2

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


def extract_features(frame, model):
    #image = Image.open(image_path)
    image = preprocess(convert_to_PIL(frame)).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    features = features.squeeze().numpy()  # Remove batch dimension and convert to numpy
    
    return features

def extract_features_Dino(frame, processor, model, device):
    
    image1 = convert_to_PIL(frame)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        features = outputs1.last_hidden_state
        features = features.mean(dim=1)
                
        return features


def similarity_Dino(feature_init, frame, processor, model, device):
    # Example image paths
    #image_paths = ['outputs/all_crops/crop_100.jpg', 'outputs/all_crops/crop_754.jpg','outputs/all_crops/crop_339.jpg', 'outputs/all_crops/crop_647.jpg','outputs/all_crops/crop_648.jpg','outputs/all_crops/crop_649.jpg']

    # Extract features for all images
    feature_frame = extract_features_Dino(frame,processor,model,device)
    #print(feature_frame.shape, feature_init.shape)
    # Compute cosine similarities between the feature vectors
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(feature_init[0],feature_frame[0]).item()
    sim = (sim+1)/2
    #print(sim)
    
    return sim

def similarity(feature_init, frame):
    # Example image paths

    # Extract features for all images
    feature_frame = extract_features(frame)

    # Compute cosine similarities between the feature vectors
    similarities = cosine_similarity([feature_init],[feature_frame])[0][0]
    
    return similarities