import open_clip
import torch
import numpy as np
from PIL import Image
import albumentations as A
import joblib
import os
import sys
from collections import Counter

# Load OpenCLIP model with specified weights
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",  # OpenCLIP weights
        device=device
    )
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Error loading CLIP model: {str(e)}")
    raise

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize classifiers and encoders as None
ingredient_classifier = None
ingredient_encoder = None
vessel_classifier = None
vessel_encoder = None

# Try to load the models, but handle missing files gracefully
try:
    # Load ingredient models
    ingredient_classifier_path = os.path.join(current_dir, "prep_ingredient_classifier.pkl")
    ingredient_encoder_path = os.path.join(current_dir, "prep_ingredient_encoder.pkl")
    
    if os.path.exists(ingredient_classifier_path) and os.path.exists(ingredient_encoder_path):
        ingredient_classifier = joblib.load(ingredient_classifier_path)
        ingredient_encoder = joblib.load(ingredient_encoder_path)
        print("Ingredient classifier and encoder loaded successfully")
        print(f"Number of ingredient classes: {len(ingredient_encoder.classes_)}")
    else:
        print("Warning: Ingredient model files not found")
    
    # Load vessel models
    vessel_classifier_path = os.path.join(current_dir, "prep_vessel_classifier.pkl")
    vessel_encoder_path = os.path.join(current_dir, "prep_vessel_encoder.pkl")
    
    if os.path.exists(vessel_classifier_path) and os.path.exists(vessel_encoder_path):
        vessel_classifier = joblib.load(vessel_classifier_path)
        vessel_encoder = joblib.load(vessel_encoder_path)
        print("Vessel classifier and encoder loaded successfully")
        print(f"Number of vessel classes: {len(vessel_encoder.classes_)}")
    else:
        print("Warning: Vessel model files not found")
    
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Check if models are loaded
if ingredient_classifier is None or ingredient_encoder is None or vessel_classifier is None or vessel_encoder is None:
    print("Warning: One or more model files are missing.")
    print("Please run the training script first: python varandah_prep_vessel_training.py")
    if __name__ == "__main__":
        sys.exit(1)

# Enhanced augmentation for test-time augmentation
augment = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.ColorJitter(p=0.2),
    A.Resize(224, 224)  # Simple resize to CLIP input size
])

def extract_features(image_path, n_aug=5, debug=False):
    """
    Extract features from an image with test-time augmentation
    
    Args:
        image_path: Path to the image file
        n_aug: Number of augmentations for test-time augmentation
        debug: Whether to print debug information
        
    Returns:
        numpy.ndarray: Feature vector or None if extraction fails
    """
    try:
        # Input validation
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if debug:
            print(f"Processing image: {image_path}")
            
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Check image dimensions
        if debug:
            print(f"Image shape: {image_np.shape}")
            
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            if debug:
                print(f"Warning: Unusual image shape {image_np.shape}, attempting to fix")
            # Try to fix the image
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 1:  # Single channel
                image_np = np.concatenate([image_np, image_np, image_np], axis=2)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                image_np = image_np[:, :, :3]
                
        # Ensure image is large enough
        if image_np.shape[0] < 10 or image_np.shape[1] < 10:
            raise ValueError(f"Image too small: {image_np.shape}")
            
        all_features = []
        
        # Multiple augmentation predictions
        for i in range(n_aug):
            try:
                # Apply augmentation
                augmented = augment(image=image_np)["image"]
                
                # Convert to PIL and preprocess for CLIP
                pil_image = Image.fromarray(augmented)
                
                # Use CLIP's preprocess function directly
                preprocessed = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Generate embedding
                with torch.no_grad():
                    features = model.encode_image(preprocessed).cpu().numpy()
                
                # Reshape if needed
                if features.shape[0] == 1 and len(features.shape) > 2:
                    features = features.reshape(1, -1)
                
                # Check for NaN values
                if np.isnan(features).any():
                    if debug:
                        print(f"Warning: NaN values in features for augmentation {i}, skipping")
                    continue
                
                all_features.append(features)
                
                if debug:
                    print(f"Augmentation {i}: shape={features.shape}")
                
            except Exception as e:
                if debug:
                    print(f"Error during augmentation {i}: {str(e)}")
                continue
        
        # Check if we have any valid features
        if not all_features:
            if debug:
                print("No valid features generated")
            return None
        
        # Return the average features across all augmentations
        return np.mean(all_features, axis=0)
        
    except Exception as e:
        if debug:
            print(f"Error extracting features: {str(e)}")
            import traceback
            print(traceback.format_exc())
        return None

def predict_class(features, classifier, encoder, confidence_threshold=0.05, debug=False):
    """
    Predict class from features
    
    Args:
        features: Feature vector
        classifier: Trained classifier
        encoder: Label encoder
        confidence_threshold: Minimum confidence required for a valid prediction
        debug: Whether to print debug information
        
    Returns:
        tuple: (predicted_class, confidence) or ("unknown", confidence) if below threshold
    """
    try:
        if features is None:
            return None, 0.0
            
        # Get prediction probabilities
        proba = classifier.predict_proba(features)
        max_idx = np.argmax(proba)
        max_confidence = proba[0][max_idx]
        
        # Only return prediction if confidence is above threshold
        if max_confidence < confidence_threshold:
            return "unknown", max_confidence
        
        # Use the class with highest confidence
        predicted_class = encoder.inverse_transform([max_idx])[0]
        
        # Get top 3 predictions and their confidences
        top3_indices = np.argsort(proba[0])[-3:][::-1]
        top3_classes = encoder.inverse_transform(top3_indices)
        top3_confidences = proba[0][top3_indices]
        
        # Print top 3 predictions
        if debug:
            print("\nTop 3 predictions:")
            for cls, conf in zip(top3_classes, top3_confidences):
                print(f"{cls}: {conf:.2%}")
        
        return predicted_class, max_confidence
        
    except Exception as e:
        if debug:
            print(f"Error during prediction: {str(e)}")
        return None, 0.0

def predict_ingredient_and_vessel(image_path, n_aug=5, confidence_threshold=0.05, debug=False):
    """
    Predict both ingredient and vessel from a single image with enhanced reliability
    
    Args:
        image_path: Path to the image file
        n_aug: Number of augmentations for test-time augmentation
        confidence_threshold: Minimum confidence required for a valid prediction
        debug: Whether to print debug information
        
    Returns:
        dict: Dictionary with ingredient and vessel predictions and confidences
    """
    # Check if models are loaded
    if ingredient_classifier is None or ingredient_encoder is None or vessel_classifier is None or vessel_encoder is None:
        print("Error: Models not loaded. Please run the training script first.")
        return {"ingredient": None, "ingredient_confidence": 0.0, 
                "vessel": None, "vessel_confidence": 0.0}
    
    try:
        # Extract features with test-time augmentation
        features = extract_features(image_path, n_aug, debug)
        if features is None:
            return {"ingredient": None, "ingredient_confidence": 0.0, 
                    "vessel": None, "vessel_confidence": 0.0}
        
        # Predict ingredient
        ingredient_name, ingredient_confidence = predict_class(
            features, ingredient_classifier, ingredient_encoder, confidence_threshold, debug
        )
        
        # Predict vessel
        vessel_name, vessel_confidence = predict_class(
            features, vessel_classifier, vessel_encoder, confidence_threshold, debug
        )
        
        return {
            "ingredient": ingredient_name,
            "ingredient_confidence": ingredient_confidence,
            "vessel": vessel_name,
            "vessel_confidence": vessel_confidence
        }
    
    except Exception as e:
        if debug:
            print(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        return {"ingredient": None, "ingredient_confidence": 0.0, 
                "vessel": None, "vessel_confidence": 0.0}

# For backward compatibility
def predict_ingredient(image_path, n_aug=5, confidence_threshold=0.05, debug=False):
    """Legacy function to only predict ingredient"""
    result = predict_ingredient_and_vessel(image_path, n_aug, confidence_threshold, debug)
    return result["ingredient"], result["ingredient_confidence"]

# For backward compatibility
def predict_vessel(image_path, n_aug=5, confidence_threshold=0.05, debug=False):
    """Legacy function to only predict vessel"""
    result = predict_ingredient_and_vessel(image_path, n_aug, confidence_threshold, debug)
    return result["vessel"], result["vessel_confidence"]

def batch_predict(image_paths, n_aug=5, confidence_threshold=0.05, debug=False):
    """
    Predict ingredients and vessels for multiple images
    
    Args:
        image_paths: List of paths to image files
        n_aug: Number of augmentations for test-time augmentation
        confidence_threshold: Minimum confidence required for a valid prediction
        debug: Whether to print debug information
        
    Returns:
        list: List of prediction dictionaries
    """
    results = []
    for image_path in image_paths:
        prediction = predict_ingredient_and_vessel(image_path, n_aug, confidence_threshold, debug)
        results.append(prediction)
    return results

if __name__ == "__main__":
    # Check if models are loaded before attempting prediction
    if ingredient_classifier is None or ingredient_encoder is None or vessel_classifier is None or vessel_encoder is None:
        print("Models not loaded. Please run the training script first:")
        print("python varandah_prep_vessel_training.py")
        sys.exit(1)
        
    # Test image path
    image_path = "/Users/atharvabadkas/Coding /CLIP/testing_pipeline/20250301/DT20250301_TM195047_MCD83BDA894D90_WT4935_TC41_TX40_RN976_DW4797.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Warning: Test image not found at {image_path}")
        # Try to find any image file to test with
        test_dir = os.path.dirname(image_path)
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            image_files = [f for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = os.path.join(test_dir, image_files[0])
                print(f"Using alternative test image: {image_path}")
        else:
            print("No test directory found. Please provide a valid image path.")
            sys.exit(1)
    
    # Get predictions with more augmentations and debug info
    predictions = predict_ingredient_and_vessel(image_path, n_aug=10, confidence_threshold=0.05, debug=True)
    
    # Print results
    print("\n=== PREDICTION RESULTS ===")
    
    if predictions["ingredient"] is None:
        print("Failed to predict ingredient")
    elif predictions["ingredient"] == "unknown":
        print(f"Ingredient prediction uncertain (confidence: {predictions['ingredient_confidence']:.2%})")
    else:
        print(f"Ingredient: {predictions['ingredient']} | Confidence: {predictions['ingredient_confidence']:.2%}")
    
    if predictions["vessel"] is None:
        print("Failed to predict vessel")
    elif predictions["vessel"] == "unknown":
        print(f"Vessel prediction uncertain (confidence: {predictions['vessel_confidence']:.2%})")
    else:
        print(f"Vessel: {predictions['vessel']} | Confidence: {predictions['vessel_confidence']:.2%}")
