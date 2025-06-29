from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from PIL import Image
import os

# Define paths
image_folder = "my_images/"  # folder containing images
output_file = "output.txt"  # file to write results

# Load model and feature extractor
feature_extractor = ViTFeatureExtractor \
    .from_pretrained("checkpoint/")
model = ViTForImageClassification \
    .from_pretrained("checkpoint/")

# Open output file
with open(output_file, 'w') as f:
    # Iterate through all image files in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            try:
                # Load image
                image = Image.open(image_path)

                # Preprocess and model inference
                inputs = feature_extractor(images=image, 
                                           return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1)[0].item()
                predicted_label = model.config.id2label[
                    predicted_class_idx]

                # Write result to file
                f.write(f"{filename}: {predicted_label}\n")
                print(f"{filename} classified as {predicted_label}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

