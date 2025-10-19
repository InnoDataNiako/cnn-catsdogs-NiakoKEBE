import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Transforms pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image, device):
    """
    Prédit la classe d'une image
    
    Args:
        model: Modèle PyTorch
        image: Image PIL
        device: CPU ou CUDA
    
    Returns:
        prediction (str), confidence (float), probabilities (dict)
    """
    # Prétraiter l'image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Classes
    classes = ['Cat 🐱', 'Dog 🐶']
    prediction = classes[predicted.item()]
    confidence_value = confidence.item()
    
    # Probabilités pour chaque classe
    probs = {
        'Cat 🐱': probabilities[0][0].item(),
        'Dog 🐶': probabilities[0][1].item()
    }
    
    return prediction, confidence_value, probs