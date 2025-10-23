import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import base64
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Yemeni Currency class mapping
CURRENCY_CLASSES = {
    0: '10 back',
    1: '10 front', 
    2: '100 back',
    3: '100 front',
    4: '1000 back',
    5: '1000 front',
    6: '200 back',
    7: '200 front',
    8: '5 back',
    9: '5 front',
    10: '500 back',
    11: '500 front'
}

def load_model():
    try:
        # Load the state dict first to check its structure
        checkpoint = torch.load('Yemeni_currency_classifier.pth', map_location='cpu')
        
        print(f"Loaded checkpoint type: {type(checkpoint)}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Initialize a plain ResNet18 (without our wrapper)
        model = models.resnet18(pretrained=False)
        
        # Modify the final fully connected layer for our number of classes
        model.fc = nn.Linear(model.fc.in_features, len(CURRENCY_CLASSES))
        
        # Remove 'module.' prefix if model was trained with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove 'module.' if present
            new_state_dict[name] = v
        
        # Load the state dict
        model.load_state_dict(new_state_dict)
        model.eval()
        
        print("Yemeni Currency ResNet18 model loaded successfully!")
        print(f"Number of classes: {len(CURRENCY_CLASSES)}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Print more details about the error
        import traceback
        traceback.print_exc()
        raise e

model = load_model()

# CORRECTED Image preprocessing - MUST MATCH YOUR TRAINING TRANSFORMATIONS
def preprocess_image(image):
    # Use the EXACT same transformations as your validation/test set
    transform = transforms.Compose([
        transforms.Resize(256),           # First resize to 256
        transforms.CenterCrop(224),       # Then center crop to 224x224
        transforms.ToTensor(),            # Convert to tensor
        transforms.Normalize(             # Same normalization as training
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read and preprocess image
            image = Image.open(file.stream).convert('RGB')
            
            # Debug: Print image info
            print(f"Original image size: {image.size}, mode: {image.mode}")
            
            input_tensor = preprocess_image(image)
            
            # Debug: Print tensor info
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(outputs[0]).item()
                confidence = probabilities[predicted_class].item()
                
                # Debug: Print raw outputs
                print(f"Raw outputs: {outputs[0]}")
                print(f"Probabilities: {probabilities}")
                print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
            
            # Apply confidence threshold (20%)
            if confidence < 0.2:
                predicted_label = "Unrecognized object"
                confidence_display = "Low confidence"
            else:
                predicted_label = CURRENCY_CLASSES.get(predicted_class, 'Unknown')
                confidence_display = f"{round(confidence * 100, 2)}%"
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'prediction': predicted_label,
                'confidence': confidence_display,
                'image_data': f"data:image/jpeg;base64,{img_str}",
                'raw_confidence': round(confidence * 100, 2)
            })
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Extract base64 image data
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Debug: Print image info
        print(f"Camera image size: {image.size}, mode: {image.mode}")
        
        # Preprocess and predict
        input_tensor = preprocess_image(image)
        
        # Debug: Print tensor info
        print(f"Camera input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs[0]).item()
            confidence = probabilities[predicted_class].item()
            
            # Debug: Print raw outputs
            print(f"Camera raw outputs: {outputs[0]}")
            print(f"Camera predicted class: {predicted_class}, Confidence: {confidence:.4f}")
        
        # Apply confidence threshold (20%)
        if confidence < 0.2:
            predicted_label = "Unrecognized object"
            confidence_display = "Low confidence"
        else:
            predicted_label = CURRENCY_CLASSES.get(predicted_class, 'Unknown')
            confidence_display = f"{round(confidence * 100, 2)}%"
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence_display,
            'raw_confidence': round(confidence * 100, 2)
        })
        
    except Exception as e:
        print(f"Camera prediction error: {str(e)}")
        return jsonify({'error': f'Camera prediction failed: {str(e)}'}), 500

@app.route('/test_model', methods=['GET'])
def test_model():
    """Test endpoint to verify model is working"""
    try:
        # Create a dummy tensor with the same shape as expected input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        return jsonify({
            'status': 'Model is working!',
            'output_shape': list(output.shape),
            'model_type': 'ResNet18',
            'currency_types': 'Yemeni (5, 10, 100, 200, 500, 1000) - Front & Back',
            'preprocessing': 'Resize(256) -> CenterCrop(224) -> Normalize(ImageNet stats)',
            'confidence_threshold': '20%'
        })
    except Exception as e:
        return jsonify({'error': f'Model test failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5005)