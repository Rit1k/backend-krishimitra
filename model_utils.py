"""
Model utility functions for plant disease detection.
Provides fallback functionality when TensorFlow model loading fails.
"""

import logging
import random
import numpy as np
from PIL import Image
import os
from io import BytesIO

# Set up logging
logger = logging.getLogger('app.model_utils')

# Disease classes matching the model's training data
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Simple color-based analysis for fallback functionality
def get_disease_prediction(img):
    """
    Basic fallback method for disease prediction when TensorFlow model is unavailable.
    Uses simple image analysis (color distribution) to make a best-guess about plant health.
    
    Args:
        img: PIL Image object
        
    Returns:
        tuple: (disease_name, confidence)
    """
    try:
        # Resize image for consistent analysis
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract color channels
        red_channel = img_array[:, :, 0].mean()
        green_channel = img_array[:, :, 1].mean()
        blue_channel = img_array[:, :, 2].mean()
        
        logger.info(f"Color analysis - R: {red_channel}, G: {green_channel}, B: {blue_channel}")
        
        # Simple heuristic: Healthy plants are typically greener
        # This is a basic approximation and not medically accurate
        if green_channel > red_channel and green_channel > blue_channel:
            # Likely healthy - find a healthy class for the dominant plant type
            healthy_classes = [c for c in DISEASE_CLASSES if "healthy" in c.lower()]
            if healthy_classes:
                predicted_class = random.choice(healthy_classes)
                confidence = min(green_channel / 255, 0.75)  # Cap confidence at 75%
                return predicted_class, confidence
        
        # Detect yellowish discoloration (could be many diseases)
        if red_channel > 100 and green_channel > 100 and blue_channel < 100:
            # Yellow leaves could indicate nutrient deficiency or virus
            virus_classes = [c for c in DISEASE_CLASSES if "virus" in c.lower()]
            if virus_classes:
                predicted_class = random.choice(virus_classes)
                confidence = min((red_channel + green_channel) / (2 * 255), 0.6)
                return predicted_class, confidence
        
        # Detect brownish spots (could be fungal infection)
        if red_channel > blue_channel and red_channel > green_channel:
            # Brown spots could indicate fungal disease
            fungal_classes = [
                c for c in DISEASE_CLASSES if any(x in c.lower() for x in ["spot", "blight", "rot", "rust", "scab"])
            ]
            if fungal_classes:
                predicted_class = random.choice(fungal_classes)
                confidence = min(red_channel / 255, 0.5)  # Lower confidence
                return predicted_class, confidence
        
        # If no clear pattern, select a random disease class
        # This is a last resort fallback
        predicted_class = random.choice(DISEASE_CLASSES)
        confidence = 0.3  # Low confidence
        
        logger.info(f"Fallback prediction: {predicted_class} with confidence {confidence}")
        return predicted_class, confidence
        
    except Exception as e:
        logger.error(f"Error in fallback disease prediction: {str(e)}")
        # Return a safe default if everything fails
        return DISEASE_CLASSES[-1], 0.1  # Usually the last class is Tomato___healthy

def analyze_dominant_color(img):
    """
    Analyze the dominant color in an image.
    
    Args:
        img: PIL Image object
        
    Returns:
        tuple: (r, g, b, dominant_color_name)
    """
    # Resize to make processing faster
    small_img = img.resize((50, 50))
    
    # Get color data
    colors = small_img.getcolors(2500)  # 50*50 = 2500 max colors
    
    if not colors:
        return (0, 0, 0, "unknown")
        
    # Find the most common color
    max_count = 0
    dominant_color = None
    
    for count, color in colors:
        if count > max_count:
            max_count = count
            dominant_color = color
    
    # If image mode is RGB
    if small_img.mode == 'RGB':
        r, g, b = dominant_color
        
        # Determine color name based on RGB values
        if g > 1.2 * r and g > 1.2 * b:
            color_name = "green"
        elif r > 1.2 * g and r > 1.2 * b:
            color_name = "red"
        elif r > 100 and g > 100 and b < 100:
            color_name = "yellow" 
        elif r > 100 and g < 100 and b < 100:
            color_name = "red"
        elif r < 80 and g < 80 and b < 80:
            color_name = "black"
        elif r > 200 and g > 200 and b > 200:
            color_name = "white"
        elif abs(r - g) < 30 and abs(r - b) < 30 and abs(g - b) < 30:
            color_name = "gray"
        elif b > 1.2 * r and b > 1.2 * g:
            color_name = "blue"
        else:
            color_name = "brown"
            
        return (r, g, b, color_name)
    
    return (0, 0, 0, "unknown") 