# Force TensorFlow to use CPU only
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS         
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
import logging.handlers
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import sys
import traceback
import requests
import json
from datetime import datetime
import numpy as np
import io
from PIL import Image
import platform
import google.generativeai as genai
import time
import re
import base64
from bson.objectid import ObjectId
import random

# Import our model utilities for plant disease detection
import model_utils




# Initialize Flask application
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": os.getenv('CORS_ORIGINS', '*').split(',')}})

# Configure Flask app for better API handling
app.url_map.strict_slashes = False
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Create logs directory if it doesn't exist
try:
    if not os.path.exists('logs'):
        os.makedirs('logs')
except Exception as e:
    print(f"Error creating logs directory: {str(e)}")

# Configure logging
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

# Add file handler for app-specific logs
try:
    file_handler = logging.FileHandler('logs/krishimitra.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up file handler: {str(e)}")

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# Start MongoDB connection
db = None
try:
    MONGODB_URI = os.getenv('MONGODB_URI')
    if MONGODB_URI:
        client = MongoClient(MONGODB_URI)
        db = client.get_database()
        logger.info("MongoDB connection established")
    else:
        logger.warning("MONGODB_URI environment variable not set, running without database")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")

# Function to ensure all environment variables are loaded
def load_environment_variables():
    """Ensure all environment variables are loaded from .env file"""
    try:
        # Try to import dotenv and load variables
        try:
            from dotenv import load_dotenv
            # Load .env file if it exists
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            if os.path.exists(env_path):
                logger.info(f"Loading environment variables from: {env_path}")
                load_dotenv(dotenv_path=env_path)
                logger.info("Environment variables loaded successfully from .env file")
            else:
                logger.warning(f".env file not found at {env_path}")
        except ImportError:
            logger.warning("python-dotenv package not installed, skipping .env loading")
        
        # Log all available environment variables (names only, not values)
        env_vars = [var for var in os.environ.keys() if not var.startswith("_")]
        logger.info(f"Available environment variables: {', '.join(env_vars)}")
        
        # Check for critical API keys
        critical_vars = ['GEMINI_API_KEY', 'OPENAI_API_KEY', 'OPENWEATHER_API_KEY', 'MONGODB_URI']
        missing_vars = [var for var in critical_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing critical environment variables: {', '.join(missing_vars)}")
        else:
            logger.info("All critical environment variables are present")
            
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")

# Load environment variables at startup
load_environment_variables()

# Initialize default variables
TF_AVAILABLE = False
model = None
GENAI_AVAILABLE = False
OPENAI_AVAILABLE = False
TWILIO_AVAILABLE = False
ALT_PREDICTOR_AVAILABLE = False

# Disease classes (update these according to your model's classes)
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

# Disease treatments (you can expand this dictionary)
DISEASE_TREATMENTS = {
    'Apple___Apple_scab': 'Apply fungicides containing captan or myclobutanil. Remove infected leaves and maintain good air circulation.',
    'Apple___Black_rot': 'Apply fungicides with captan or thiophanate-methyl. Remove infected fruit and prune out diseased branches and cankers.',
    'Apple___Cedar_apple_rust': 'Apply fungicides containing myclobutanil or propiconazole. Remove nearby cedar trees if possible to break the disease cycle.',
    'Apple___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate nutrition, and regular monitoring.',
    'Blueberry___healthy': 'No treatment needed. Maintain good cultural practices including proper soil pH (4.5-5.5), adequate mulching, and regular monitoring.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply sulfur-based fungicides or potassium bicarbonate. Prune to improve air circulation and avoid overhead irrigation.',
    'Cherry_(including_sour)___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate nutrition, and regular monitoring.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides containing pyraclostrobin or azoxystrobin. Practice crop rotation and consider resistant varieties.',
    'Corn_(maize)___Common_rust': 'Apply fungicides containing azoxystrobin or propiconazole. Plant resistant hybrids and time planting to avoid peak rust periods.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Apply fungicides with pyraclostrobin or propiconazole. Plant resistant hybrids and implement crop rotation.',
    'Corn_(maize)___healthy': 'No treatment needed. Maintain good cultural practices including crop rotation, proper fertility, and weed management.',
    'Grape___Black_rot': 'Apply fungicides containing myclobutanil or mancozeb. Remove mummified berries and prune out infected wood.',
    'Grape___Esca_(Black_Measles)': 'No effective chemical treatment. Prune in dry weather, disinfect tools, and apply wound sealant. Severely infected vines may need removal.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides with mancozeb or copper compounds. Improve air circulation and avoid overhead irrigation.',
    'Grape___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, canopy management, and regular monitoring.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure available. Control Asian citrus psyllid with insecticides. Remove and destroy infected trees to prevent spread.',
    'Peach___Bacterial_spot': 'Apply copper-based bactericides. Plant resistant varieties and maintain good air circulation through proper pruning.',
    'Peach___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate nutrition, and regular monitoring.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper-based bactericides. Use pathogen-free seeds and practice crop rotation. Avoid overhead irrigation.',
    'Pepper,_bell___healthy': 'No treatment needed. Maintain good cultural practices including proper spacing, adequate nutrition, and regular monitoring.',
    'Potato___Early_blight': 'Apply fungicides containing chlorothalonil or azoxystrobin. Practice crop rotation and provide adequate plant spacing.',
    'Potato___Late_blight': 'Apply fungicides containing chlorothalonil, mancozeb, or metalaxyl. Destroy volunteer plants and ensure good air circulation.',
    'Potato___healthy': 'No treatment needed. Maintain good cultural practices including proper hilling, adequate nutrition, and regular monitoring.',
    'Raspberry___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate spacing, and regular monitoring.',
    'Soybean___healthy': 'No treatment needed. Maintain good cultural practices including crop rotation, proper fertility, and weed management.',
    'Squash___Powdery_mildew': 'Apply fungicides containing sulfur or potassium bicarbonate. Space plants for good air circulation and avoid overhead irrigation.',
    'Strawberry___Leaf_scorch': 'Apply fungicides containing captan or myclobutanil. Renovate beds annually and provide adequate spacing between plants.',
    'Strawberry___healthy': 'No treatment needed. Maintain good cultural practices including proper spacing, mulching, and regular monitoring.',
    'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Use pathogen-free seeds and practice crop rotation. Remove and destroy infected plant material.',
    'Tomato___Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Remove lower leaves showing symptoms and mulch soil to prevent splash-up.',
    'Tomato___Late_blight': 'Apply fungicides containing chlorothalonil, mancozeb, or copper compounds. Improve air circulation and use drip irrigation.',
    'Tomato___Leaf_Mold': 'Apply fungicides containing chlorothalonil or copper compounds. Reduce humidity and improve air circulation. Avoid leaf wetness.',
    'Tomato___Septoria_leaf_spot': 'Apply fungicides containing chlorothalonil or copper compounds. Remove infected leaves promptly and apply mulch to prevent soil splash.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply insecticidal soap or horticultural oil. For severe infestations, use miticides. Keep plants well-watered to prevent stress.',
    'Tomato___Target_Spot': 'Apply fungicides containing chlorothalonil or azoxystrobin. Improve air circulation and remove infected plant material.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure once infected. Remove infected plants and control whitefly vectors with appropriate insecticides.',
    'Tomato___Tomato_mosaic_virus': 'No cure once infected. Remove and destroy infected plants. Use virus-free certified seed and disinfect tools between plants.',
    'Tomato___healthy': 'No treatment needed. Maintain good cultural practices including proper staking, pruning, and watering at the base of plants.'
}

# Special handling for Windows platform and DLLs
if platform.system() == 'Windows':
    logger.info("Windows platform detected, setting up DLL directories")
    try:
        # Add system directories to DLL search path to help with TensorFlow import on Windows
        os.add_dll_directory("C:/Windows/System32")
    except Exception as e:
        logger.error(f"Error setting up DLL directories: {str(e)}")
else:
    # Non-Windows platforms don't need special DLL handling
    pass

# Try to import TensorFlow and set up the model
if not (platform.system() == 'Windows' and 'tf' in locals() and TF_AVAILABLE):
    try:
        # Force TensorFlow to use CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Lower TensorFlow's logging level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        logger.info("Attempting to import TensorFlow...")
        import tensorflow as tf
        
        # Add specific TensorFlow imports
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        
        # Get TensorFlow version
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Set TensorFlow logging level to suppress warnings
        tf.get_logger().setLevel('ERROR')
        
        TF_AVAILABLE = True
        logger.info("TensorFlow successfully imported (CPU mode)")
    except ImportError as e:
        logger.warning(f"TensorFlow could not be imported - disease detection will be unavailable. Error: {str(e)}")
        TF_AVAILABLE = False
        model = None
    except Exception as e:
        logger.error(f"Error initializing TensorFlow: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        TF_AVAILABLE = False
        model = None


@app.route('/upload-model', methods=['POST'])
def upload_model():
    global model

    if 'model' not in request.files:
        return 'No model file provided', 400

    model_file = request.files['model']
    os.makedirs('model', exist_ok=True)  # Ensure directory exists
    model_path = os.path.join('model', 'plant_disease_model.h5')

    try:
        model_file.save(model_path)
        print(f"✅ Model saved to {model_path}")

        # Load the model immediately after saving
        model = load_model(model_path)
        print("✅ Model loaded successfully")

        return 'Model uploaded and loaded successfully', 200

    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return f'Error uploading model: {e}', 500


# Try to load the disease detection model
if TF_AVAILABLE:
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5')
        if os.path.exists(model_path):
            logger.info(f"Loading plant disease model from: {model_path}")
            try:
                # Force model to load on CPU with custom options for better compatibility
                with tf.device('/CPU:0'):
                    # Use a custom object scope to handle any custom layers/objects
                    tf_model = tf.keras.models.load_model(
                        model_path,
                        compile=False,  # Don't compile the model to avoid additional issues
                        custom_objects=None  # Allow TensorFlow to handle any custom objects automatically
                    )
                
                # Log model architecture summary to help with debugging
                model_summary = []
                tf_model.summary(print_fn=lambda x: model_summary.append(x))
                logger.info("Model summary:\n" + "\n".join(model_summary))
                
                # Log device placement and other important info
                logger.info("Model device placement: CPU only (forced)")
                logger.info(f"Model input shape: {tf_model.input_shape}")
                logger.info(f"Model output shape: {tf_model.output_shape}")
                
                # Adjust DISEASE_CLASSES if needed to match model output
                if hasattr(tf_model, 'output_shape') and tf_model.output_shape[1] != len(DISEASE_CLASSES):
                    logger.warning(f"Model output shape ({tf_model.output_shape[1]}) doesn't match number of disease classes ({len(DISEASE_CLASSES)})")
                    
                    if tf_model.output_shape[1] == 38 and len(DISEASE_CLASSES) == 38:
                        logger.info("Model output shape and disease classes count both match 38 classes")
                    else:
                        logger.warning("Consider updating DISEASE_CLASSES to match model output dimension")
                else:
                    logger.info("Model output shape matches number of disease classes")
                
                # Make a test prediction to verify model is functioning
                test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                test_input[:, :, 1] = 0.8  # Green channel
                try:
                    with tf.device('/CPU:0'):
                        test_output = tf_model.predict(test_input, verbose=0)
                    logger.info(f"Test prediction successful, output shape: {test_output.shape}")
                except Exception as pred_error:
                    logger.error(f"Test prediction failed: {str(pred_error)}")
                    
            except Exception as model_error:
                logger.error(f"Error loading model: {str(model_error)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                tf_model = None
                # Continue running even without the model
                logger.warning("Continuing without plant disease model - disease detection will be disabled")
        else:
            logger.warning(f"Model file not found at: {model_path}")
            tf_model = None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        tf_model = None
        logger.warning("Continuing without plant disease model - disease detection will be disabled")
else:
    logger.warning("TensorFlow not available, skipping model loading")
    tf_model = None

# Try to import the alternative model predictor
ALT_PREDICTOR_AVAILABLE = False
try:
    import alt_model
    alt_predictor_test = alt_model.get_model_prediction(Image.new('RGB', (224, 224), color=(0, 200, 0)))
    if alt_predictor_test and len(alt_predictor_test) == 2:
        ALT_PREDICTOR_AVAILABLE = True
        logger.info("Alternative model predictor is available and functional")
    else:
        logger.warning("Alternative model predictor available but returned invalid result")
except ImportError:
    logger.warning("Alternative model predictor not available")
except Exception as e:
    logger.error(f"Error initializing alternative model predictor: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")

# Try to import and configure other APIs - also made optional
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    logger.info("Google GenerativeAI successfully imported")
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("Google GenerativeAI could not be imported - chat functionality will be limited")

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI successfully imported")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI could not be imported - may limit fallback functionality")

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
    logger.info("Twilio successfully imported")
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio could not be imported - SMS functionality will be unavailable")

# API Keys and Configurations
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# Configure Gemini API
if GEMINI_API_KEY:
    try:
        logger.info(f"Attempting to configure Gemini API with key: {GEMINI_API_KEY[:5]}...")
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
        
        # Test the API configuration
        try:
            # Use gemini-2.0-flash model
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content("Test connection")
            if response and hasattr(response, 'text') and response.text:
                logger.info("Gemini API test successful")
                GENAI_AVAILABLE = True
            else:
                logger.error("Gemini API test returned empty response")
                GEMINI_API_KEY = None
                GENAI_AVAILABLE = False
        except Exception as test_error:
            logger.error(f"Gemini API test failed: {str(test_error)}")
            GEMINI_API_KEY = None
            GENAI_AVAILABLE = False
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {str(e)}")
        GEMINI_API_KEY = None
        GENAI_AVAILABLE = False
else:
    logger.warning("Gemini API key not found in environment variables")
    # Try to load it from .env file directly as a backup
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            logger.info("Found Gemini API key in .env file, attempting to configure...")
            genai.configure(api_key=gemini_key)
            
            # Test the API configuration
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                # Test if the model works
                test_response = model.generate_content("Hello, test")
                logger.info("Fallback model initialized successfully")
                GENAI_AVAILABLE = True
                GEMINI_API_KEY = gemini_key
            except Exception as fallback_error:
                logger.error(f"Fallback model initialization failed: {str(fallback_error)}")
                GENAI_AVAILABLE = False
        else:
            logger.error("No Gemini API key found in .env file")
            GENAI_AVAILABLE = False
    except ImportError:
        logger.warning("python-dotenv not available, skipping direct .env loading")
        GENAI_AVAILABLE = False
    except Exception as env_error:
        logger.error(f"Error trying to load Gemini API key from .env: {str(env_error)}")
        GENAI_AVAILABLE = False


# Log available API keys (without exposing the actual keys)
logger.info(f"OpenAI API key available: {bool(OPENAI_API_KEY)}")
logger.info(f"Gemini API key available: {bool(GEMINI_API_KEY)}")
logger.info(f"OpenWeather API key available: {bool(WEATHER_API_KEY)}")
logger.info(f"Twilio account SID available: {bool(TWILIO_ACCOUNT_SID)}")
logger.info(f"Twilio auth token available: {bool(TWILIO_AUTH_TOKEN)}")
logger.info(f"Google Maps API key available: {bool(GOOGLE_MAPS_API_KEY)}")

# Initialize APIs
twilio_client = None

# Initialize Twilio if available
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_AVAILABLE:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialized")
    except Exception as e:
        logger.error(f"Error initializing Twilio client: {str(e)}")

def preprocess_image(image, target_size=(224, 224), preprocessing_method='standard'):
    """Preprocess the image for model prediction using different methods."""
    try:
        # Log original image details
        logger.info(f"Preprocessing image with method: {preprocessing_method}")
        
        # First convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Converted image to RGB mode")
        
        # Resize the image - always use the exact size the model was trained on
        if preprocessing_method == 'lanczos':
            image = image.resize(target_size, Image.LANCZOS)
            logger.info(f"Resized image using LANCZOS method to {target_size}")
        else:
            image = image.resize(target_size)
            logger.info(f"Resized image using default method to {target_size}")
        
        # Convert to array - using float32 for precision
        img_array = np.array(image, dtype=np.float32)
        logger.info(f"Converted image to array, shape: {img_array.shape}")
        
        # Normalize based on preprocessing method
        if preprocessing_method == 'imagenet':
            # ImageNet normalization
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            img_array = (img_array - mean) / std
            logger.info("Applied ImageNet normalization")
        elif preprocessing_method == 'zero_center':
            # Zero-center normalization (-1 to 1 range)
            img_array = (img_array - 127.5) / 127.5
            logger.info("Applied zero-center normalization [-1,1]")
        else:
            # Standard [0,1] normalization - this is most common for trained models
            img_array = img_array / 255.0
            logger.info(f"Applied standard normalization [0,1], new range: {img_array.min():.4f} to {img_array.max():.4f}")
        
        # Check for NaN values which can cause prediction issues
        if np.isnan(img_array).any() or np.isinf(img_array).any():
            logger.warning("NaN or Inf values detected in preprocessed image")
            img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Log shape for debugging
        logger.info(f"Final preprocessed image shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def analyze_image_colors(img):
    """
    Analyze image colors for a fallback prediction method when ML model fails.
    Returns disease prediction based on color characteristics.
    """
    logger.info("Starting image color analysis fallback")
    
    try:
        # Resize for consistent analysis
        img_resized = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Calculate color channel means
        red_mean = np.mean(img_array[:,:,0])
        green_mean = np.mean(img_array[:,:,1])
        blue_mean = np.mean(img_array[:,:,2])
        
        # Calculate color ratios
        green_to_red_ratio = green_mean / max(red_mean, 1)
        green_to_blue_ratio = green_mean / max(blue_mean, 1)
        
        # Calculate color statistics
        color_std = np.std(img_array)
        green_std = np.std(img_array[:,:,1])
        
        # Log color information
        logger.info(f"Color analysis - R: {red_mean:.2f}, G: {green_mean:.2f}, B: {blue_mean:.2f}")
        logger.info(f"Color ratios - G/R: {green_to_red_ratio:.2f}, G/B: {green_to_blue_ratio:.2f}")
        logger.info(f"Color variation - StdDev: {color_std:.2f}, Green StdDev: {green_std:.2f}")
        
        # Disease prediction logic based on color characteristics
        disease_name = None
        confidence = 0.0
        
        # Check for healthy plant (strong green dominant)
        if green_mean > red_mean * 1.3 and green_mean > blue_mean * 1.3 and green_std < 50: 
            disease_name = "Tomato___healthy"
            confidence = min(0.7, green_to_red_ratio / 3)
            logger.info("Color analysis suggests healthy plant (strong green)")
            
        # Check for leaf spot diseases (higher red/brown, lower green)
        elif red_mean > green_mean and color_std > 60:
            disease_name = "Tomato___Bacterial_spot"
            confidence = min(0.6, (red_mean / green_mean) / 2)
            logger.info("Color analysis suggests bacterial spot (red/brown spots)")
            
        # Check for yellowing (higher red+green, lower blue)
        elif red_mean > 100 and green_mean > 100 and blue_mean < 80 and abs(red_mean - green_mean) < 40:
            disease_name = "Tomato___Yellow_Leaf_Curl_Virus"
            confidence = min(0.5, ((red_mean + green_mean) / (2 * blue_mean)) / 4)
            logger.info("Color analysis suggests Yellow Leaf Curl Virus (yellowing)")
            
        # Check for early blight (brownish patches)
        elif red_mean > 80 and green_mean < 90 and blue_mean < 70:
            disease_name = "Tomato___Early_blight"
            confidence = min(0.5, (red_mean / (green_mean + blue_mean)) / 3)
            logger.info("Color analysis suggests Early Blight (brownish patches)")
            
        # Check for late blight (dark patches with some green)
        elif red_mean < 100 and green_mean < 100 and blue_mean < 100 and color_std > 50:
            disease_name = "Tomato___Late_blight"
            confidence = min(0.5, color_std / 200)
            logger.info("Color analysis suggests Late Blight (dark patches)")
            
        # Default case - select the most common disease with low confidence
        else:
            disease_name = "Tomato___Leaf_Mold"
            confidence = 0.4
            logger.info("Color analysis inconclusive, using fallback disease")
        
        # Get treatment for the detected disease
        treatment = DISEASE_TREATMENTS.get(disease_name, "No specific treatment information available. Please consult an agricultural expert.")
        
        return disease_name, confidence, treatment
        
    except Exception as e:
        logger.error(f"Error in image color analysis: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Last resort fallback
        return "Tomato___healthy", 0.3, "Could not determine treatment. Please consult an agricultural expert."

@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    global model
    """Predict plant disease from uploaded image."""
    
    # Initialize variables
    prediction_method = "unknown"
    warning_message = None
    
    if 'image' not in request.files:
        logger.error("No image file part in the request")
        return jsonify({"error": "No image file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    # Check if model is loaded
    if not TF_AVAILABLE or tf_model is None:
        warning_message = "Warning: TensorFlow model not available. Using basic image analysis fallback."
        logger.warning(warning_message)
    
    try:
        # Open and validate image
        try:
            img = Image.open(file.stream)
            logger.info(f"Original image format: {img.format}, size: {img.size}, mode: {img.mode}")
            
            # Convert to RGB if not already
            if img.mode != "RGB":
                img = img.convert("RGB")
                logger.info(f"Converted image to RGB mode")
        except Exception as img_error:
            logger.error(f"Error opening image: {str(img_error)}")
            return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400
        
        # Initialize result variables
        disease_name = None
        confidence = 0.0
        treatment = None
        top_predictions = []
        
        # Try model-based prediction if available
        if TF_AVAILABLE and tf_model is not None:
            try:
                logger.info("Using TensorFlow model for prediction")
                
                # Use standard preprocessing - most likely what the model was trained with
                img_array = preprocess_image(img, preprocessing_method='standard')
                
                # Make prediction with model
                logger.info("Making model prediction with standard preprocessing")
                start_time = time.time()
                with tf.device('/CPU:0'):
                    predictions = tf_model.predict(img_array, verbose=0)
                prediction_time = time.time() - start_time
                logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
                logger.info(f"Raw prediction output shape: {predictions.shape}")
                
                # Log raw prediction values for debugging
                if len(predictions) > 0:
                    top_5_indices = np.argsort(predictions[0])[::-1][:5]
                    for i, idx in enumerate(top_5_indices):
                        if idx < len(DISEASE_CLASSES):
                            logger.info(f"Top {i+1} prediction: {DISEASE_CLASSES[idx]} with value {predictions[0][idx]:.4f}")
                
                # Process predictions
                if len(predictions) > 0 and len(predictions[0]) > 0:
                    # Get sorted indices of top predictions
                    sorted_indices = np.argsort(predictions[0])[::-1]
                    
                    # Get top 5 predictions or all if less than 5
                    top_n = min(5, len(sorted_indices))
                    top_indices = sorted_indices[:top_n]
                    
                    # Create list of top predictions
                    for idx in top_indices:
                        if idx < len(DISEASE_CLASSES):
                            prediction_confidence = float(predictions[0][idx])
                            disease = DISEASE_CLASSES[idx]
                            top_predictions.append({
                                "disease": disease,
                                "confidence": prediction_confidence
                            })
                    
                    # Get the top prediction (highest confidence)
                    top_idx = sorted_indices[0]
                    
                    # Safety check for index bounds
                    if top_idx < len(DISEASE_CLASSES):
                        disease_name = DISEASE_CLASSES[top_idx]
                        confidence = float(predictions[0][top_idx])
                        treatment = DISEASE_TREATMENTS.get(disease_name, "No specific treatment information available.")
                        prediction_method = "tensorflow_model"
                        
                        logger.info(f"Final model prediction: {disease_name} with confidence {confidence:.4f}")
                    else:
                        logger.error(f"Prediction index {top_idx} is out of bounds for DISEASE_CLASSES (length: {len(DISEASE_CLASSES)})")
                        warning_message = "Warning: Model output index is out of bounds. Using best-effort result."
                else:
                    logger.error("Model produced empty prediction array")
                    warning_message = "Warning: Model produced no predictions. Using fallback analysis."
            except Exception as pred_error:
                logger.error(f"Model prediction error: {str(pred_error)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                warning_message = "Warning: Error in model prediction. Using fallback analysis."
        
        # If model prediction failed or unavailable, use image analysis fallback
        if disease_name is None or confidence == 0.0:
            logger.info("Using image analysis fallback for prediction")
            
            # Call color analysis fallback
            try:
                disease_name, confidence, treatment = analyze_image_colors(img)
                prediction_method = "color_analysis"
                
                # Create a single entry for top predictions
                top_predictions = [{
                    "disease": disease_name,
                    "confidence": confidence
                }]
                
                logger.info(f"Color analysis prediction: {disease_name} with confidence {confidence:.4f}")
            except Exception as fallback_error:
                logger.error(f"Fallback analysis error: {str(fallback_error)}")
                return jsonify({
                    "error": "Failed to generate prediction with any available method",
                    "details": str(fallback_error)
                }), 500
        
        # Prepare response with top predictions
        response = {
            "disease": disease_name,
            "confidence": confidence,
            "treatment": treatment,
            "method": prediction_method,
            "top_predictions": top_predictions
        }
        
        # Add warning if there was one
        if warning_message:
            response["warning"] = warning_message
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unhandled exception in detect_disease: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/api/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}'
    response = requests.get(url)
    return jsonify(response.json())

@app.route('/api/transporters', methods=['GET'])
def get_transporters():
    try:
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        
        if not lat or not lon:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        # Search for various transport-related places
        transport_types = [
            'moving_company',
            'storage',
            'truck_rental',
            'transit_station',
            'warehouse'
        ]
        
        all_results = []
        
        for place_type in transport_types:
            url = (
                f'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
                f'?location={lat},{lon}'
                f'&radius=10000'  # 10km radius
                f'&type={place_type}'
                f'&key={GOOGLE_MAPS_API_KEY}'
            )
            
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') == 'OK':
                all_results.extend(data.get('results', []))
            elif data.get('status') == 'ZERO_RESULTS':
                continue
            else:
                logger.error(f"Error from Google Places API: {data.get('status')} - {data.get('error_message', 'No error message')}")
                if data.get('status') == 'REQUEST_DENIED':
                    return jsonify({'error': 'API key is invalid or missing'}), 401
                elif data.get('status') == 'OVER_QUERY_LIMIT':
                    return jsonify({'error': 'API quota exceeded'}), 429
        
        # Remove duplicates based on place_id
        seen_places = set()
        unique_results = []
        for place in all_results:
            if place['place_id'] not in seen_places:
                seen_places.add(place['place_id'])
                unique_results.append(place)
        
        return jsonify({
            'results': unique_results,
            'count': len(unique_results)
        })
    
    except requests.RequestException as e:
        logger.error(f"Network error while fetching transport services: {str(e)}")
        return jsonify({'error': 'Failed to fetch transport services'}), 503
    except Exception as e:
        logger.error(f"Unexpected error in get_transporters: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        user_id = data.get('user_id', 'anonymous')
        is_subscribed = data.get('is_subscribed', False)
        
        logger.info(f"Chat request - User ID: {user_id}, Is subscribed (client): {is_subscribed}")
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Check subscription status for message limits
        user = None
        user_is_subscribed = False
        
        if user_id != 'anonymous' and db:
            # First, look for the direct user document by ID
            user = db.users.find_one({'_id': user_id})
            
            # If not found, try user_id as a field
            if not user:
                user = db.users.find_one({'user_id': user_id})
                
            # Check if this user is related to a subscribed phone or email
            if not user:
                # Try to find any subscription with this user ID
                subscribed_user = db.users.find_one({
                    '$or': [
                        {'phone_user_id': user_id},
                        {'email_user_id': user_id}
                    ],
                    'subscribed': True
                })
                
                if subscribed_user:
                    user_is_subscribed = True
            else:
                user_is_subscribed = user.get('subscribed', False)
                
            logger.info(f"User found: {user is not None}, DB subscription status: {user_is_subscribed}")
            
            # If client says they're subscribed or database confirms subscription
            if is_subscribed or user_is_subscribed:
                user_is_subscribed = True
                logger.info(f"User {user_id} has an active subscription")
        
        # For non-subscribers, check message count
        if db and not user_is_subscribed:
            message_count = db.messages.count_documents({'user_id': user_id})
            logger.info(f"Message count for user {user_id}: {message_count}")
            
            # Increase the free message limit to 10 messages
            if message_count >= 10:
                return jsonify({
                    'error': 'Message limit reached',
                    'response': "You've reached the limit of 10 free messages. To continue chatting, please subscribe to our service. You can subscribe by providing your email or phone number in the subscription form.",
                    'limit_reached': True,
                    'message_count': message_count
                }), 403
        
        # First check if Gemini API is available and configured
        if not GENAI_AVAILABLE:
            logger.error("Google GenerativeAI module is not available")
            return generate_fallback_response(user_message, user_id)
            
        # Then check if Gemini API key is configured
        global GEMINI_API_KEY
        if not GEMINI_API_KEY:
            logger.error("Gemini API key not configured or invalid")
            
            # Try to reconfigure with environment variable
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                try:
                    logger.info("Attempting to reconfigure Gemini API with environment variable...")
                    genai.configure(api_key=gemini_key)
                    model_test = genai.GenerativeModel('gemini-2.0-flash')
                    test_response = model_test.generate_content("Test connection")
                    if test_response and hasattr(test_response, 'text') and test_response.text:
                        logger.info("Gemini API reconfigured successfully")
                        GEMINI_API_KEY = gemini_key
                    else:
                        logger.error("Gemini API reconfiguration test failed")
                        return generate_fallback_response(user_message, user_id)
                except Exception as reconfig_error:
                    logger.error(f"Gemini API reconfiguration failed: {str(reconfig_error)}")
                    return generate_fallback_response(user_message, user_id)
            else:
                logger.error("No Gemini API key available in environment variables")
                return generate_fallback_response(user_message, user_id)
        
        try:
            # Initialize Gemini AI
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Get user context if available
            user_context = ""
            if user_id != 'anonymous' and db:
                # Fetch last 5 messages to provide context
                previous_messages = list(db.messages.find(
                    {'user_id': user_id},
                    {'message': 1, 'response': 1, '_id': 0}
                ).sort('timestamp', -1).limit(5))
                
                if previous_messages:
                    user_context = "Previous conversation:\n"
                    for msg in reversed(previous_messages):
                        user_context += f"User: {msg.get('message')}\n"
                        user_context += f"Assistant: {msg.get('response')}\n"
            
            # Prepare improved agricultural context
            agricultural_prompt = f"""You are KrishiMitra, an AI agricultural assistant for Indian farmers.

Role: Provide helpful, accurate, and practical advice about farming in India.

Context: You are speaking with a farmer in India who has the following question or concern:
"{user_message}"

{user_context}

Guidelines:
1. Focus on sustainable farming practices appropriate for India
2. Provide region-specific advice when possible
3. Suggest local crop varieties and techniques
4. Give practical solutions that would be feasible for farmers in India
5. Consider the unique challenges of Indian agriculture (monsoon dependence, small holdings, etc.)
6. Be respectful and helpful, acknowledging the farmer's knowledge
7. When appropriate, suggest both traditional methods and modern innovations
8. For plant diseases, mention both organic and chemical treatments
9. If uncertain about location-specific information, ask clarifying questions

IMPORTANT: If you don't know something specific to Indian farming conditions, acknowledge this honestly and suggest general principles or reliable sources where they can find more information.

Please respond in a clear, helpful manner using simple language that's easy to understand."""
            
            # Define safety settings to ensure appropriate responses
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Generate response with retry mechanism
            max_retries = 3
            retry_count = 0
            ai_response = None
            
            while retry_count < max_retries and not ai_response:
                try:
                    logger.info(f"Attempting to generate response (attempt {retry_count + 1}/{max_retries})")
                    
                    # For gemini-2.0-flash model
                    response = model.generate_content(
                        agricultural_prompt,
                        safety_settings=safety_settings,
                        generation_config={
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40,
                            "max_output_tokens": 800,
                        }
                    )
                    
                    if response and hasattr(response, 'text') and response.text:
                        ai_response = response.text
                        logger.info("Successfully generated response from Gemini API")
                    else:
                        logger.warning("Empty response from Gemini API")
                        retry_count += 1
                        time.sleep(1)  # Add a short delay before retry
                        
                except Exception as retry_error:
                    error_message = str(retry_error)
                    logger.error(f"Error on attempt {retry_count + 1}: {error_message}")
                    
                    # Add specific error handling for common Gemini API errors
                    if "400 Request payload is invalid" in error_message:
                        logger.error("Invalid request payload. Simplifying prompt...")
                        # Try a simpler prompt on the next attempt
                        agricultural_prompt = f"You are KrishiMitra, an agricultural assistant for Indian farmers. Please answer this question: {user_message}"
                    elif "429" in error_message:
                        logger.error("Rate limit exceeded. Waiting longer before retry...")
                        time.sleep(3)  # Wait longer for rate limit errors
                        
                    retry_count += 1
                    time.sleep(1)  # Brief pause between retries
            
            # If all retries failed, fall back to rule-based response
            if not ai_response:
                logger.error("All Gemini API attempts failed")
                return generate_fallback_response(user_message, user_id)
            
            # Log the conversation to database if available and check subscription status at the same time
            updated_subscription_status = log_conversation(user_id, user_message, ai_response)
            
            # If subscription status has changed, update our local variable
            if updated_subscription_status:
                user_is_subscribed = True
            
            return jsonify({
                'response': ai_response,
                'message_count': message_count if 'message_count' in locals() else 0,
                'subscription_status': 'active' if user_is_subscribed else 'inactive'
            })
            
        except Exception as ai_error:
            logger.error(f"Error with Gemini AI: {str(ai_error)}")
            return generate_fallback_response(user_message, user_id)
            
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'response': "I'm sorry, something went wrong. Please try again later."
        }), 500

def generate_fallback_response(user_message, user_id):
    # First try with OpenAI if available as a backup
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            logger.info("Attempting to use OpenAI as fallback")
            openai.api_key = OPENAI_API_KEY
            
            # Create a simple prompt for OpenAI
            prompt = f"""You are KrishiMitra, an AI assistant helping Indian farmers.

User query: {user_message}

Provide a helpful, accurate response about farming in India. Focus on practical advice."""
            
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                ai_response = response['choices'][0]['text'].strip()
                logger.info("Successfully generated OpenAI fallback response")
                
                # Log the conversation to database if available
                log_conversation(user_id, user_message, ai_response)
                
                return jsonify({
                    'response': ai_response,
                    'source': 'openai_fallback'
                })
                
        except Exception as openai_error:
            logger.error(f"OpenAI fallback failed: {str(openai_error)}")
            # Continue to rule-based fallback
    
    # Simple rule-based fallback responses for common agricultural topics
    query = user_message.lower()
    
    # Enhanced rule-based fallbacks for common agricultural topics
    if "weather" in query or "rain" in query or "forecast" in query:
        fallback_message = "For accurate weather information, please use our Weather section where you can get real-time forecasts for your location. If you'd like to discuss weather patterns and their impact on your crops, please try again when our AI service is back online."
    
    elif "disease" in query or "pest" in query or "infection" in query:
        fallback_message = "For plant disease identification, please use our Disease Detection feature where you can upload a photo of your plant. This will give you more accurate results than text descriptions. We can analyze the image and suggest appropriate treatments."
    
    elif "fertilizer" in query or "nutrient" in query or "soil" in query:
        fallback_message = "For soil health and fertilizer advice, I recommend checking soil pH and nutrient levels before application. Organic options include compost, manure, and crop rotation. NPK fertilizers should be applied according to crop-specific needs and growth stage."
    
    elif "market" in query or "price" in query or "sell" in query:
        fallback_message = "For market information and prices, I recommend checking local agricultural markets (mandis) or using government resources like eNAM (Electronic National Agriculture Market) for current rates. Consider forming farmer producer organizations for better bargaining power."
    
    elif "transport" in query or "shipping" in query or "delivery" in query:
        fallback_message = "Our Transport section can help you find reliable transportation services for your produce. You can check available options, compare rates, and arrange pickups directly through our platform."
    
    elif "seed" in query or "variety" in query or "planting" in query:
        fallback_message = "For seed selection, consider both traditional and high-yielding varieties appropriate for your region. Purchase from reliable government or certified sources to ensure quality. Factors to consider include local climate, soil type, disease resistance, and market demand."
    
    elif "organic" in query or "natural" in query or "chemical-free" in query:
        fallback_message = "Organic farming practices include crop rotation, composting, biological pest control, and natural fertilizers. Certification requires following specific standards for 3+ years. Benefits include premium pricing and sustainability, though yields may initially be lower than conventional methods."
    
    elif "irrigation" in query or "water" in query or "drought" in query:
        fallback_message = "Efficient irrigation methods include drip systems (saving 30-50% water), sprinklers, and traditional methods like furrow irrigation. Consider rainwater harvesting, mulching to retain moisture, and drought-resistant crop varieties in water-scarce regions."
    
    elif "subsidy" in query or "government" in query or "scheme" in query:
        fallback_message = "Major government programs include PM-KISAN (income support), Soil Health Card, crop insurance (PMFBY), interest subsidies, and various state-specific schemes. Contact your local agriculture office or visit official government portals for application details."
    
    else:
        # General agricultural advice for any other query
        fallback_message = "I understand you're asking about agricultural topics. While our AI service is currently limited, KrishiMitra offers comprehensive farming information through our platform. For specific needs, please use our Disease Detection feature for plant health, Weather section for forecasts, and Transport section for logistics. Our team is continuously working to improve our services to better assist Indian farmers."
    
    # Log the conversation to database if available
    log_conversation(user_id, user_message, fallback_message)
    
    return jsonify({
        'response': fallback_message,
        'warning': 'Using fallback response system'
    })

def log_conversation(user_id, user_message, ai_response):
    # Log the conversation to database if available
    try:
        if db:
            # First check if this user is subscribed
            subscribed = False
            user = None
            
            # Check direct _id match
            user = db.users.find_one({'_id': user_id})
            
            # If not found, check for user_id field
            if not user:
                user = db.users.find_one({'user_id': user_id})
            
            # If still not found, check phone_user_id or email_user_id
            if not user:
                user = db.users.find_one({
                    '$or': [
                        {'phone_user_id': user_id},
                        {'email_user_id': user_id}
                    ]
                })
            
            # If user found, get subscription status
            if user:
                subscribed = user.get('subscribed', False)
                logger.info(f"User {user_id} found in database, subscription status: {subscribed}")
            
            # Log the message with subscription status
            db.messages.insert_one({
                'user_id': user_id,
                'message': user_message,
                'response': ai_response,
                'timestamp': datetime.utcnow(),
                'subscribed': subscribed
            })
            
            return subscribed
    except Exception as db_error:
        logger.error(f"Error logging chat message to database: {str(db_error)}")
        # Continue even if logging fails
    
    return False

@app.route('/api/check-subscription', methods=['GET'])
def check_subscription():
    """Check if a phone number or email is already subscribed"""
    logger.info("API call to /api/check-subscription")
    
    phone = request.args.get('phone')
    email = request.args.get('email')
    
    if not phone and not email:
        logger.warning("No phone or email provided for subscription check")
        return jsonify({'subscribed': False, 'error': 'No phone or email provided'}), 400
    
    logger.info(f"Checking subscription for phone: {phone}, email: {email}")
    
    # Check if database is available
    if not db:
        logger.warning("MongoDB not available, cannot check subscription")
        return jsonify({'subscribed': False, 'error': 'Database unavailable'}), 503
    
    try:
        query = {}
        if phone:
            # Handle 10-digit Indian phone number
            cleaned_phone = ''.join(filter(str.isdigit, phone))
            if len(cleaned_phone) == 10:
                formatted_phone = '+91' + cleaned_phone
                query['phone'] = formatted_phone
            else:
                query['phone'] = phone
        
        if email:
            query['email'] = email
        
        # If both phone and email provided, check either
        if phone and email:
            user = db.users.find_one({'$or': [{'phone': query['phone']}, {'email': query['email']}]})
        else:
            user = db.users.find_one(query)
        
        logger.info(f"User found: {user is not None}")
        
        if user:
            is_subscribed = user.get('subscribed', False)
            logger.info(f"User subscription status: {is_subscribed}")
            return jsonify({
                'subscribed': is_subscribed,
                'phone': phone,
                'email': email,
                'subscription_details': {
                    'timestamp': user.get('timestamp'),
                    'source': user.get('source')
                } if is_subscribed else None
            })
        else:
            logger.info(f"No subscription found for phone: {phone}, email: {email}")
            return jsonify({'subscribed': False, 'phone': phone, 'email': email})
    
    except Exception as e:
        logger.error(f"Error checking subscription: {str(e)}")
        return jsonify({'subscribed': False, 'error': str(e)}), 500

@app.route('/api/subscribers/email', methods=['GET'])
def get_email_subscribers():
    try:
        # Get all users with email_subscribed set to True
        subscribers = list(db.users.find(
            {'email_subscribed': True}, 
            {'email': 1, 'name': 1, 'subscription_date': 1, 'source': 1, '_id': 0}
        ))
        
        logger.info(f"Found {len(subscribers)} email subscribers")
        return jsonify(subscribers)
    except Exception as e:
        logger.error(f"Error getting email subscribers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribers/sms', methods=['GET'])
def get_sms_subscribers():
    try:
        # Get all users with sms_subscribed set to True
        subscribers = list(db.users.find(
            {'sms_subscribed': True}, 
            {'phone': 1, 'name': 1, 'subscription_date': 1, 'source': 1, '_id': 0}
        ))
        
        logger.info(f"Found {len(subscribers)} SMS subscribers")
        return jsonify(subscribers)
    except Exception as e:
        logger.error(f"Error getting SMS subscribers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribers/email', methods=['DELETE'])
def unsubscribe_email():
    try:
        email = request.json.get('email')
        if not email:
            return jsonify({'error': 'Email is required'}), 400
            
        # Delete the user with this email
        result = db.users.delete_one({'email': email})
        
        if result.deleted_count == 0:
            logger.warning(f"No user found with email {email}")
            return jsonify({'error': 'No user found with this email'}), 404
            
        logger.info(f"User with email {email} deleted from database")
        return jsonify({'success': True, 'message': f'User with email {email} unsubscribed successfully'})
    except Exception as e:
        logger.error(f"Error unsubscribing email: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribers/sms', methods=['DELETE'])
def unsubscribe_sms():
    try:
        phone = request.json.get('phone')
        if not phone:
            return jsonify({'error': 'Phone number is required'}), 400
            
        # Delete the user with this phone
        result = db.users.delete_one({'phone': phone})
        
        if result.deleted_count == 0:
            logger.warning(f"No user found with phone {phone}")
            return jsonify({'error': 'No user found with this phone number'}), 404
            
        logger.info(f"User with phone {phone} deleted from database")
        return jsonify({'success': True, 'message': f'User with phone {phone} unsubscribed successfully'})
    except Exception as e:
        logger.error(f"Error unsubscribing SMS: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    """Subscribe or unsubscribe a user with phone or email"""
    try:
        data = request.json
        phone = data.get('phone')
        email = data.get('email')
        name = data.get('name', '')
        source = data.get('source', 'website')
        unsubscribe = data.get('unsubscribe', False)  # Added unsubscribe parameter
        
        if not phone and not email:
            return jsonify({'error': 'Phone number or email is required'}), 400
            
        # For phone subscriptions
        if phone:
            # Check if user already exists
            existing_user = db.users.find_one({'phone': phone})
            
            # Handle unsubscribe request
            if unsubscribe:
                if existing_user:
                    db.users.delete_one({'phone': phone})
                    logger.info(f"Deleted SMS subscriber {phone} from database")
                    return jsonify({'success': True, 'message': 'SMS unsubscription successful'})
                else:
                    return jsonify({'error': 'Phone number not found in subscribers list'}), 404
            
            # Check if already subscribed
            if existing_user and existing_user.get('sms_subscribed', False):
                logger.info(f"Phone {phone} is already subscribed")
                return jsonify({'error': 'Phone number already subscribed'}), 409
            
            # Create or update phone subscription
            if existing_user:
                # Update existing user
                db.users.update_one(
                    {'phone': phone},
                    {'$set': {
                        'sms_subscribed': True,
                        'name': name or existing_user.get('name', ''),
                        'source': source,
                        'subscription_date': datetime.utcnow()
                    }}
                )
                logger.info(f"Updated SMS subscription for {phone}")
            else:
                # Create new user
                db.users.insert_one({
                    'phone': phone,
                    'name': name,
                    'sms_subscribed': True,
                    'source': source,
                    'subscription_date': datetime.utcnow()
                })
                logger.info(f"Added new SMS subscriber: {phone}")
                
            return jsonify({'success': True, 'message': 'SMS subscription successful'})
        
        # For email subscriptions
        elif email:
            # Check if user already exists
            existing_user = db.users.find_one({'email': email})
            
            # Handle unsubscribe request
            if unsubscribe:
                if existing_user:
                    db.users.delete_one({'email': email})
                    logger.info(f"Deleted email subscriber {email} from database")
                    return jsonify({'success': True, 'message': 'Email unsubscription successful'})
                else:
                    return jsonify({'error': 'Email not found in subscribers list'}), 404
            
            # Check if already subscribed
            if existing_user and existing_user.get('email_subscribed', False):
                logger.info(f"Email {email} is already subscribed")
                return jsonify({'error': 'Email already subscribed'}), 409
            
            # Create or update email subscription
            if existing_user:
                # Update existing user
                db.users.update_one(
                    {'email': email},
                    {'$set': {
                        'email_subscribed': True,
                        'name': name or existing_user.get('name', ''),
                        'source': source,
                        'subscription_date': datetime.utcnow()
                    }}
                )
                logger.info(f"Updated email subscription for {email}")
            else:
                # Create new user
                db.users.insert_one({
                    'email': email,
                    'name': name,
                    'email_subscribed': True,
                    'source': source,
                    'subscription_date': datetime.utcnow()
                })
                logger.info(f"Added new email subscriber: {email}")
                
            return jsonify({'success': True, 'message': 'Email subscription successful'})
            
    except Exception as e:
        logger.error(f"Error in subscription process: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-bulk-email', methods=['POST'])
def send_bulk_email():
    """Send email to all subscribers"""
    try:
        # Get data from request
        data = request.json
        subject = data.get('subject')
        message_content = data.get('message')
        
        if not all([subject, message_content]):
            return jsonify({
                'error': 'Missing required parameters',
                'message': 'Subject and message are required'
            }), 400
        
        # Check database connection
        if not db:
            return jsonify({'error': 'Database not available'}), 503
        
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_username = os.getenv('SMTP_USERNAME')
        smtp_password = os.getenv('SMTP_PASSWORD')
        sender_email = os.getenv('SENDER_EMAIL', smtp_username)
        
        if not all([smtp_server, smtp_username, smtp_password]):
            return jsonify({
                'error': 'Email service unavailable',
                'message': 'Email service is not configured properly'
            }), 503
        
        # Get all subscribed emails
        subscribers = list(db.users.find(
            {'email': {'$exists': True}, 'subscribed': True}, 
            {'email': 1}
        ))
        
        if not subscribers:
            return jsonify({
                'warning': 'No subscribers found',
                'message': 'No email subscribers to send message to'
            }), 200
        
        # Send email to each subscriber
        sent_count = 0
        error_count = 0
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            
            for subscriber in subscribers:
                recipient_email = subscriber.get('email')
                if not recipient_email:
                    error_count += 1
                    continue
                
                try:
                    # Create message
                    email_message = MIMEMultipart()
                    email_message['From'] = sender_email
                    email_message['To'] = recipient_email
                    email_message['Subject'] = subject
                    
                    # Add HTML body with unsubscribe link
                    body = f"""
                    <html>
                    <body>
                        {message_content}
                        <p style="font-size: 12px; color: #666; margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px;">
                            This email was sent to {recipient_email} because you subscribed to KrishiMitra updates.
                            <br>
                            If you no longer wish to receive these emails, you can <a href="http://localhost:5000?unsubscribe={recipient_email}">unsubscribe here</a>.
                        </p>
                    </body>
                    </html>
                    """
                    
                    email_message.attach(MIMEText(body, 'html'))
                    
                    # Send email
                    server.send_message(email_message)
                    sent_count += 1
                    
                except Exception as email_error:
                    logger.error(f"Error sending email to {recipient_email}: {str(email_error)}")
                    error_count += 1
        
        logger.info(f"Bulk email sent to {sent_count} subscribers with {error_count} errors")
        return jsonify({
            'success': True,
            'message': f'Bulk email sent successfully',
            'sent': sent_count,
            'errors': error_count,
            'total': len(subscribers)
        })
    
    except Exception as e:
        logger.error(f"Error in send_bulk_email endpoint: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

# Setting up flags to inform the frontend about available features
FEATURES = {
    "disease_detection": True,  # Always available with model_utils
    "chatbot": OPENAI_AVAILABLE or GENAI_AVAILABLE,
    "weather": WEATHER_API_KEY is not None,
    "sms": TWILIO_AVAILABLE
}
logger.info(f"Available features: {FEATURES}")

# Add a new endpoint to check available features
@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify(FEATURES)

@app.route('/api/send-email', methods=['POST'])
def send_email():
    """Send email to a specific subscriber"""
    try:
        # Get data from request
        data = request.json
        recipient_email = data.get('email')
        subject = data.get('subject')
        message_content = data.get('message')
        
        if not all([recipient_email, subject, message_content]):
            return jsonify({
                'error': 'Missing required parameters',
                'message': 'Email, subject, and message are required'
            }), 400
        
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_username = os.getenv('EMAIL_USERNAME')
        smtp_password = os.getenv('EMAIL_PASSWORD')
        sender_email = os.getenv('EMAIL_FROM', smtp_username)
        
        if not all([smtp_server, smtp_username, smtp_password]):
            return jsonify({
                'error': 'Email service unavailable',
                'message': 'Email service is not configured properly'
            }), 503
        
        # Create message
        email_message = MIMEMultipart()
        email_message['From'] = sender_email
        email_message['To'] = recipient_email
        email_message['Subject'] = subject
        
        # Add HTML body
        email_message.attach(MIMEText(message_content, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(email_message)
        
        logger.info(f"Email sent successfully to {recipient_email}")
        return jsonify({
            'success': True,
            'message': 'Email sent successfully'
        })
    
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return jsonify({
            'error': 'Failed to send email',
            'message': str(e)
        }), 500

@app.route('/api/send-sms', methods=['POST'])
def send_sms():
    """Send SMS to a specific phone number"""
    try:
        data = request.json
        phone = data.get('phone')
        message = data.get('message')
        
        if not phone or not message:
            return jsonify({
                'error': 'Missing required parameters',
                'message': 'Both phone number and message are required'
            }), 400
        
        if not TWILIO_AVAILABLE or not twilio_client:
            return jsonify({
                'error': 'SMS service unavailable',
                'message': 'Twilio service is not configured or unavailable'
            }), 503
        
        # Send SMS using Twilio
        try:
            sms = twilio_client.messages.create(
                body=message,
                from_=os.getenv('TWILIO_PHONE'),
                to=phone
            )
            
            return jsonify({
                'success': True,
                'message': 'SMS sent successfully',
                'sms_sid': sms.sid
            })
        except Exception as twilio_error:
            logger.error(f"Twilio error sending SMS: {str(twilio_error)}")
            return jsonify({
                'error': 'Failed to send SMS',
                'message': str(twilio_error)
            }), 500
    
    except Exception as e:
        logger.error(f"Error in send_sms endpoint: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/send-bulk-sms', methods=['POST'])
def send_bulk_sms():
    """Send SMS to all subscribed phone numbers"""
    try:
        data = request.json
        message = data.get('message')
        
        if not message:
            return jsonify({
                'error': 'Missing required parameter',
                'message': 'Message content is required'
            }), 400
        
        if not TWILIO_AVAILABLE or not twilio_client:
            return jsonify({
                'error': 'SMS service unavailable',
                'message': 'Twilio service is not configured or unavailable'
            }), 503
        
        # Get all subscribed phone numbers
        subscribers = list(db.users.find(
            {'sms_subscribed': True}, 
            {'phone': 1, 'name': 1}
        ))
        
        if not subscribers:
            return jsonify({
                'warning': 'No subscribers found',
                'message': 'No SMS subscribers to send message to'
            }), 200
        
        # Send SMS to each subscriber
        success_count = 0
        failed_count = 0
        failures = []
        
        for subscriber in subscribers:
            phone = subscriber.get('phone')
            if not phone:
                failed_count += 1
                continue
                
            try:
                twilio_client.messages.create(
                    body=message,
                    from_=os.getenv('TWILIO_PHONE'),
                    to=phone
                )
                success_count += 1
            except Exception as twilio_error:
                logger.error(f"Error sending SMS to {phone}: {str(twilio_error)}")
                failed_count += 1
                failures.append({'phone': phone, 'error': str(twilio_error)})
        
        return jsonify({
            'success': True,
            'message': f'Bulk SMS sent to {success_count} subscribers, {failed_count} failed',
            'total': len(subscribers),
            'success_count': success_count,
            'failed_count': failed_count,
            'failures': failures if failures else None
        })
    
    except Exception as e:
        logger.error(f"Error in send_bulk_sms endpoint: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/test-tensorflow', methods=['GET'])
def test_tensorflow():
    """Test if TensorFlow is working"""
    response = {
        'tensorflow_available': TF_AVAILABLE,
        'model_loaded': tf_model is not None,
        'test_prediction': None,
        'model_info': {}
    }
    
    # Add info about TensorFlow
    if TF_AVAILABLE:
        response['tensorflow_version'] = tf.__version__
        response['cpu_only'] = True  # Always true since we're forcing CPU only
        response['visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    
    # Check if model is loaded
    if tf_model is not None:
        try:
            # Add model info
            response['model_info'] = {
                'input_shape': str(tf_model.input_shape),
                'output_shape': str(tf_model.output_shape),
                'output_classes': tf_model.output_shape[1] if hasattr(tf_model, 'output_shape') else None,
                'model_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5'),
                'model_exists': os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5')),
                'disease_classes_count': len(DISEASE_CLASSES)
            }
            
            # Print first few classes to check ordering
            response['model_info']['sample_classes'] = DISEASE_CLASSES[:5]
            
            # Create a test image (green colored to simulate a plant)
            test_img = np.zeros((224, 224, 3), dtype=np.float32)
            # Make it mostly green with some variation
            test_img[:, :, 1] = 0.8  # Green channel
            test_img[:, :, 0] = 0.2  # Red channel
            test_img[:, :, 2] = 0.2  # Blue channel
            # Add random noise
            test_img += np.random.rand(224, 224, 3) * 0.1
            # Clip to ensure values stay in [0, 1]
            test_img = np.clip(test_img, 0, 1)
            # Add batch dimension
            test_img = np.expand_dims(test_img, axis=0)
            
            # Try to predict
            with tf.device('/CPU:0'):
                start_time = time.time()
                predictions = tf_model.predict(test_img, verbose=0)
                prediction_time = time.time() - start_time
            
            response['model_info']['prediction_time_ms'] = round(prediction_time * 1000, 2)
            
            # Get all predictions sorted by confidence
            top5_indices = np.argsort(predictions[0])[::-1][:5]
            top5_predictions = []
            
            for i, idx in enumerate(top5_indices):
                if idx < len(DISEASE_CLASSES):
                    top5_predictions.append({
                        'rank': i+1,
                        'disease': DISEASE_CLASSES[idx],
                        'confidence': float(predictions[0][idx])
                    })
            
            response['test_prediction'] = {
                'disease': DISEASE_CLASSES[top5_indices[0]] if top5_indices[0] < len(DISEASE_CLASSES) else f"Class_{top5_indices[0]}",
                'confidence': float(predictions[0][top5_indices[0]]),
                'top5': top5_predictions
            }
            
            # Check if model's output shape matches our disease classes
            if tf_model.output_shape[1] != len(DISEASE_CLASSES):
                response['warning'] = f"Model expects {tf_model.output_shape[1]} classes but DISEASE_CLASSES has {len(DISEASE_CLASSES)}. Predictions may be inaccurate."
            
            response['tensorflow_working'] = True
            
        except Exception as e:
            response['tensorflow_working'] = False
            response['error'] = str(e)
            response['error_type'] = type(e).__name__
            response['traceback'] = traceback.format_exc()
    
    return jsonify(response)

@app.route('/api/diagnose-model', methods=['POST'])
def diagnose_model():
    """Diagnostic endpoint to check why model predictions might be incorrect"""
    try:
        # Check if TensorFlow and model are available
        if not TF_AVAILABLE or tf_model is None:
            return jsonify({
                'error': 'TensorFlow or model not available',
                'tensorflow_available': TF_AVAILABLE,
                'model_loaded': tf_model is not None
            }), 500
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Process the image
        img_data = file.read()
        img_buffer = io.BytesIO(img_data)
        img = Image.open(img_buffer)
        
        # Get diagnostic information
        diagnosis = {
            'image_info': {
                'format': img.format,
                'size': img.size,
                'mode': img.mode
            },
            'model_info': {
                'input_shape': str(tf_model.input_shape),
                'output_shape': str(tf_model.output_shape),
                'expected_classes': tf_model.output_shape[1] if hasattr(tf_model, 'output_shape') else None,
                'disease_classes_count': len(DISEASE_CLASSES)
            },
            'preprocessing': {}
        }
        
        # Test different preprocessing methods to see which gives the best results
        # Method 1: Standard preprocessing (resize to 224x224, normalize to 0-1)
        img_std = img.copy()
        img_std = img_std.convert('RGB')
        img_std = img_std.resize((224, 224))
        x_std = np.array(img_std) / 255.0
        x_std = np.expand_dims(x_std, axis=0)
        
        # Method 2: Alternative preprocessing with LANCZOS resampling
        img_lanczos = img.copy()
        img_lanczos = img_lanczos.convert('RGB')
        img_lanczos = img_lanczos.resize((224, 224), Image.LANCZOS)
        x_lanczos = np.array(img_lanczos) / 255.0
        x_lanczos = np.expand_dims(x_lanczos, axis=0)
        
        # Method 3: Preprocessing with channel normalization (ImageNet style)
        img_norm = img.copy()
        img_norm = img_norm.convert('RGB')
        img_norm = img_norm.resize((224, 224), Image.LANCZOS)
        x_norm = np.array(img_norm) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x_norm = (x_norm - mean) / std
        x_norm = np.expand_dims(x_norm, axis=0)
        
        # Get predictions for each method
        with tf.device('/CPU:0'):
            pred_std = tf_model.predict(x_std, verbose=0)
            pred_lanczos = tf_model.predict(x_lanczos, verbose=0)
            pred_norm = tf_model.predict(x_norm, verbose=0)
        
        # Process results
        methods = {
            'standard': {
                'image': 'Standard resizing',
                'predictions': get_top_predictions(pred_std[0], 5)
            },
            'lanczos': {
                'image': 'LANCZOS resizing',
                'predictions': get_top_predictions(pred_lanczos[0], 5)
            },
            'normalized': {
                'image': 'ImageNet normalization',
                'predictions': get_top_predictions(pred_norm[0], 5)
            }
        }
        
        diagnosis['preprocessing_methods'] = methods
        
        # Add suggestions based on results
        suggestions = []
        
        if tf_model.output_shape[1] != len(DISEASE_CLASSES):
            suggestions.append(f"The model expects {tf_model.output_shape[1]} classes but your DISEASE_CLASSES list has {len(DISEASE_CLASSES)} entries. Make sure they match.")
        
        # Check if the best predictions are consistent across methods
        best_std = methods['standard']['predictions'][0]['disease'] if methods['standard']['predictions'] else None
        best_lanczos = methods['lanczos']['predictions'][0]['disease'] if methods['lanczos']['predictions'] else None
        best_norm = methods['normalized']['predictions'][0]['disease'] if methods['normalized']['predictions'] else None
        
        if best_std != best_lanczos or best_std != best_norm:
            suggestions.append("Different preprocessing methods yield different top predictions. The model may be sensitive to preprocessing. Try using the preprocessing method that matches how the model was trained.")
        
        # Add class information
        diagnosis['class_info'] = {
            'first_5_classes': DISEASE_CLASSES[:5],
            'last_5_classes': DISEASE_CLASSES[-5:] if len(DISEASE_CLASSES) >= 5 else DISEASE_CLASSES,
            'total_classes': len(DISEASE_CLASSES)
        }
        
        diagnosis['suggestions'] = suggestions
        
        return jsonify(diagnosis)
        
    except Exception as e:
        return jsonify({
            'error': 'Error during diagnosis',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

def get_top_predictions(predictions, n=5):
    """Helper function to get top N predictions"""
    indices = np.argsort(predictions)[::-1][:n]
    results = []
    
    for i, idx in enumerate(indices):
        if idx < len(DISEASE_CLASSES):
            results.append({
                'rank': i+1,
                'disease': DISEASE_CLASSES[idx],
                'confidence': float(predictions[idx])
            })
    
    return results

if __name__ == "__main__":
    # Display feature availability info
    available_features = {
        'disease_detection': TF_AVAILABLE or ALT_PREDICTOR_AVAILABLE,
        'chatbot': GENAI_AVAILABLE,
        'weather': bool(WEATHER_API_KEY),
        'sms': bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
    }
    logger.info(f"Available features: {available_features}")
    
    # Print startup info
    logger.info("Starting KrishiMitra backend server")
    
    # Run the app
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
        app.run(host="0.0.0.0", port=port)

    if __name__ == '__main__':
        model_path = 'model/plant_disease_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("✅ Model pre-loaded on server start.")
    
    app.run(debug=False, host='0.0.0.0', port=5000)