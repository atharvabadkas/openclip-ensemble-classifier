import os
import open_clip
import torch
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import tqdm
from collections import defaultdict
import sys

# Load CLIP model
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

# Enhanced Augmentation Pipeline
augment = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.ColorJitter(p=0.2),
    A.Resize(224, 224)  # Simple resize to CLIP input size
])

def generate_balanced_embeddings(dataset_path, is_vessel=False, min_samples_per_class=None, aug_per_image=10, use_all_images=True):
    """
    Generate balanced embeddings by taking equal samples from each class
    and applying multiple augmentations
    
    Args:
        dataset_path: Path to the dataset directory
        is_vessel: Whether this is a vessel dataset (affects class naming)
        min_samples_per_class: Minimum number of samples per class (None = auto-determine)
        aug_per_image: Number of augmentations to generate per image
        use_all_images: Whether to use all available images (True) or limit to min_samples_per_class (False)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
    train_embeddings = []
    train_labels = []
    class_counts = defaultdict(int)
    
    # First, count available images per class
    print("Scanning dataset directory...")
    valid_classes = []
    for cls_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, cls_folder)
        if not os.path.isdir(class_path) or cls_folder.startswith('.'):
            continue
        
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if img_files:
            class_counts[cls_folder] = len(img_files)
            valid_classes.append(cls_folder)
            print(f"Found {len(img_files)} images in class {cls_folder}")
    
    if not valid_classes:
        raise ValueError("No valid class folders found in the dataset directory")
    
    # Determine minimum samples per class if not specified
    if min_samples_per_class is None:
        min_samples = min(class_counts.values())
        min_samples_per_class = min_samples
    
    print(f"\nTotal classes: {len(valid_classes)}")
    if use_all_images:
        print(f"Using ALL available images per class")
        print(f"Generating {aug_per_image} augmentations per image")
        total_images = sum(class_counts.values())
        print(f"Total images to process: {total_images}")
    else:
        print(f"Using {min_samples_per_class} samples per class")
        print(f"Generating {aug_per_image} augmentations per image")
        total_images = min_samples_per_class * len(valid_classes)
        print(f"Total images to process: {total_images}")
    
    for cls_folder in valid_classes:
        class_path = os.path.join(dataset_path, cls_folder)
        # For vessels, use the folder name directly; otherwise, remove "_images" suffix
        cls_name = cls_folder if is_vessel else cls_folder.replace("_images", "")
        
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # If not using all images, limit to min_samples_per_class
        if not use_all_images and len(img_files) > min_samples_per_class:
            np.random.seed(42)  # For reproducibility
            img_files = np.random.choice(img_files, min_samples_per_class, replace=False)
        
        print(f"\nProcessing class {cls_name}:")
        print(f"- Using {len(img_files)} images")
        print(f"- Generating {aug_per_image} augmentations per image")
        
        class_embeddings = []
        class_labels = []
        
        for img_file in tqdm.tqdm(img_files, desc=f"Processing {cls_name}"):
            image_path = os.path.join(class_path, img_file)
            
            try:
                # Load and verify image
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
                
                if image_np.size == 0 or image_np.shape[0] == 0 or image_np.shape[1] == 0:
                    print(f"Warning: Empty image at {image_path}, skipping")
                    continue
                
                for aug_idx in range(aug_per_image):
                    try:
                        # Apply augmentation
                        augmented = augment(image=image_np)["image"]
                        
                        # Convert to PIL and preprocess for CLIP
                        pil_image = Image.fromarray(augmented)
                        preprocessed = preprocess(pil_image).unsqueeze(0).to(device)
                        
                        # Generate embedding
                        with torch.no_grad():
                            embedding = model.encode_image(preprocessed).cpu().numpy()
                        
                        # Check for NaN values
                        if np.isnan(embedding).any():
                            print(f"Warning: NaN values in embedding for {image_path}, skipping")
                            continue
                            
                        class_embeddings.append(embedding)
                        class_labels.append(cls_name)
                        
                    except Exception as aug_error:
                        print(f"Error during augmentation {aug_idx} of {image_path}: {str(aug_error)}")
                        continue
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Add class embeddings to overall dataset
        if class_embeddings:
            train_embeddings.extend(class_embeddings)
            train_labels.extend(class_labels)
            print(f"Added {len(class_embeddings)} embeddings for class {cls_name}")
        else:
            print(f"Warning: No valid embeddings generated for class {cls_name}")
    
    if not train_embeddings:
        raise ValueError("No valid embeddings generated. Check your dataset and image processing pipeline.")
    
    # Convert to numpy arrays
    try:
        embeddings_array = np.vstack([e.reshape(1, -1) for e in train_embeddings])
        labels_array = np.array(train_labels)
        print(f"Successfully created embeddings array with shape {embeddings_array.shape}")
        return embeddings_array, labels_array
    except Exception as e:
        print(f"Error creating final arrays: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to datasets
    ingredients_path = "/Users/atharvabadkas/Coding /CLIP/prep_models/img_vessel_model/verandah_prep/verandah_prep_ingredients"
    vessels_path = "/Users/atharvabadkas/Coding /CLIP/prep_models/img_vessel_model/verandah_prep/verandah_vessel_prep_images"
    
    # Verify paths exist
    if not os.path.exists(ingredients_path):
        print(f"Error: Ingredients path not found: {ingredients_path}")
        exit(1)
    
    if not os.path.exists(vessels_path):
        print(f"Error: Vessels path not found: {vessels_path}")
        exit(1)
    
    try:
        # Train ingredient classifier
        print("\n=== TRAINING INGREDIENT CLASSIFIER ===")
        print(f"Using dataset path: {ingredients_path}")
        
        # Generate balanced embeddings for ingredients
        print("\nGenerating balanced embeddings for ingredients...")
        ingredient_embeddings, ingredient_labels = generate_balanced_embeddings(
            ingredients_path,
            is_vessel=False,
            min_samples_per_class=None,  # Auto-determine based on smallest class
            aug_per_image=10,
            use_all_images=True  # Use ALL available images
        )
        
        print("\nIngredient embeddings shape:", ingredient_embeddings.shape)
        print("Number of ingredient labels:", len(ingredient_labels))
        
        # Encode ingredient labels
        print("\nEncoding ingredient labels...")
        ingredient_encoder = LabelEncoder()
        encoded_ingredient_labels = ingredient_encoder.fit_transform(ingredient_labels)
        
        # Compute class weights for ingredients
        print("\nComputing ingredient class weights...")
        unique_ingredient_classes = np.unique(encoded_ingredient_labels)
        if len(unique_ingredient_classes) < 2:
            print("Warning: Only one ingredient class found. Cannot train classifier.")
            sys.exit(1)
            
        ingredient_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_ingredient_classes,
            y=encoded_ingredient_labels
        )
        ingredient_class_weight_dict = dict(zip(unique_ingredient_classes, ingredient_class_weights))
        
        # Train SVM for ingredients with class weights
        print("\nTraining ingredient SVM classifier...")
        ingredient_classifier = SVC(
            kernel='linear',
            probability=True,
            C=0.1,
            class_weight=ingredient_class_weight_dict
        )
        ingredient_classifier.fit(ingredient_embeddings, encoded_ingredient_labels)
        
        # Save ingredient models
        print("\nSaving ingredient models...")
        joblib.dump(ingredient_classifier, os.path.join(current_dir, "prep_ingredient_classifier.pkl"))
        joblib.dump(ingredient_encoder, os.path.join(current_dir, "prep_ingredient_encoder.pkl"))
        
        print("\nIngredient classifier training completed!")
        print(f"Number of ingredient classes: {len(np.unique(ingredient_labels))}")
        print(f"Total ingredient samples: {len(ingredient_labels)}")
        print("\nIngredient class distribution:")
        for cls in np.unique(ingredient_labels):
            count = np.sum(ingredient_labels == cls)
            print(f"{cls}: {count} samples")
        
        # Train vessel classifier
        print("\n\n=== TRAINING VESSEL CLASSIFIER ===")
        print(f"Using dataset path: {vessels_path}")
        
        # Generate balanced embeddings for vessels
        print("\nGenerating balanced embeddings for vessels...")
        vessel_embeddings, vessel_labels = generate_balanced_embeddings(
            vessels_path,
            is_vessel=True,
            min_samples_per_class=None,  # Auto-determine based on smallest class
            aug_per_image=10,
            use_all_images=True  # Use ALL available images
        )
        
        print("\nVessel embeddings shape:", vessel_embeddings.shape)
        print("Number of vessel labels:", len(vessel_labels))
        
        # Encode vessel labels
        print("\nEncoding vessel labels...")
        vessel_encoder = LabelEncoder()
        encoded_vessel_labels = vessel_encoder.fit_transform(vessel_labels)
        
        # Compute class weights for vessels
        print("\nComputing vessel class weights...")
        unique_vessel_classes = np.unique(encoded_vessel_labels)
        if len(unique_vessel_classes) < 2:
            print("Warning: Only one vessel class found. Cannot train classifier.")
            sys.exit(1)
            
        vessel_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_vessel_classes,
            y=encoded_vessel_labels
        )
        vessel_class_weight_dict = dict(zip(unique_vessel_classes, vessel_class_weights))
        
        # Train SVM for vessels with class weights
        print("\nTraining vessel SVM classifier...")
        vessel_classifier = SVC(
            kernel='linear',
            probability=True,
            C=0.1,
            class_weight=vessel_class_weight_dict
        )
        vessel_classifier.fit(vessel_embeddings, encoded_vessel_labels)
        
        # Save vessel models
        print("\nSaving vessel models...")
        joblib.dump(vessel_classifier, os.path.join(current_dir, "prep_vessel_classifier.pkl"))
        joblib.dump(vessel_encoder, os.path.join(current_dir, "prep_vessel_encoder.pkl"))
        
        print("\nVessel classifier training completed!")
        print(f"Number of vessel classes: {len(np.unique(vessel_labels))}")
        print(f"Total vessel samples: {len(vessel_labels)}")
        print("\nVessel class distribution:")
        for cls in np.unique(vessel_labels):
            count = np.sum(vessel_labels == cls)
            print(f"{cls}: {count} samples")
        
        print("\nAll models saved successfully in:", current_dir)
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())