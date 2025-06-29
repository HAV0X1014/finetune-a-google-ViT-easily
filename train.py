import os
import torch
# Import torchvision for data loading and transforms
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from PIL import Image # Pillow is needed for image handling by torchvision

# --- Configuration ---
MODEL_NAME = "google/vit-base-patch16-224-in21k"
TRAIN_DATA_DIR = "./dataset/" # <--- CHANGE THIS TO YOUR TRAINING DATA PATH
VAL_DATA_DIR = "./validation/"   # <--- CHANGE THIS TO YOUR VALIDATION DATA PATH (Set to None if no validation data)
OUTPUT_DIR = "./output/"

# --- 1. Load Data ---

# Ensure data directories exist
if not os.path.exists(TRAIN_DATA_DIR):
    raise FileNotFoundError(f"Training data directory not found: {TRAIN_DATA_DIR}")

# Use torchvision's ImageFolder to load the datasets, applying the transforms
train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR)

# Get class names and mappings from the training dataset
# ImageFolder automatically sets .classes based on directory names
class_names = train_dataset.classes
label2id = {name: i for i, name in enumerate(class_names)}
id2label = {i: name for i, name in enumerate(class_names)}
num_labels = len(class_names)

print(f"Found {num_labels} classes: {class_names}")
print(f"Training data found: {len(train_dataset)} images.")

val_dataset = None
if VAL_DATA_DIR and os.path.exists(VAL_DATA_DIR):
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR)
    print(f"Validation data found: {len(val_dataset)} images.")
else:
    print(f"Warning: Validation data directory not found or not provided: {VAL_DATA_DIR}. Skipping evaluation and early stopping based on validation metrics.")


# --- 2. Load Model ---
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Load the pre-trained ViT model for image classification
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Allows loading even if the classification head size doesn't match
)

print("Model loaded.")

# --- 3. Data Collator (MODIFIED) ---

# The collate_fn needs to process the (PIL Image, label) tuples from ImageFolder.
# It collects PIL images and passes them as a list to the Hugging Face processor.
def collate_fn(batch):
    """
    This function is used by the DataLoader to process a batch.
    It takes a list of (PIL Image, label) tuples (from ImageFolder)
    and applies the Hugging Face processor's full pipeline.
    """
    # batch is a list of tuples: [(pil_img1, label1), (pil_img2, label2), ...]
    # Collect the PIL Images into a list
    images = [item[0] for item in batch]
    # Collect the labels into a list
    labels = [item[1] for item in batch]

    # Apply the Hugging Face processor to the list of PIL Images.
    # This single call handles resizing, cropping, conversion to tensor,
    # scaling [0, 255] to [0, 1], and normalization using the pre-trained mean/std.
    # return_tensors="pt" ensures PyTorch tensors are returned.
    batch_processor_output = processor(images, return_tensors="pt")

    # Add the labels to the dictionary
    batch_processor_output['labels'] = torch.tensor(labels)

    return batch_processor_output

print("Collate function defined (processing PIL images directly with HF processor).")

# --- 4. Define Metrics ---

def compute_metrics(eval_pred):
    """Computes accuracy."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predicted_labels)}

print("Metrics function defined.")

# --- 5. Configure Training Arguments (Optimized for speed and avoiding overfitting) ---

# Define training parameters. Adjust these according to your needs and resources.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                 # Output directory for checkpoints and logs
    # Set a generous number of epochs, early stopping will handle when to stop
    num_train_epochs=6,                   # Increased maximum epochs

    # --- Speed & VRAM Optimization ---
    # Adjust batch sizes based on your 24GB VRAM. Start high and decrease if OOM.
    per_device_train_batch_size=16,       # <--- Increased Batch Size per GPU
    per_device_eval_batch_size=16,        # <--- Increased Batch Size for Evaluation

    # Enable mixed precision for speed (BF16 is recommended for RDNA 3 with ROCm)
    # bf16=True,                             # <--- Enable BF16 mixed precision
    # If bf16 causes issues, try: fp16=True # Enable FP16 mixed precision

    # Increase data loader workers to feed data faster (adjust based on CPU cores)
    dataloader_num_workers=2,              # <--- Increased Data Loader Workers

    # --- Regularization and Learning Rate ---
    # Use a small learning rate for fine-tuning
    #learning_rate=5e-5,                    # <--- Fine-tuning Learning Rate (experiment between 1e-5 and 3e-5)

    warmup_steps=500,                      # Number of steps for the learning rate warmup
    weight_decay=0.01,                     # Strength of weight decay (prevents large weights)

    # --- Logging and Evaluation ---
    logging_dir=f"{OUTPUT_DIR}/logs",      # Directory for storing logs
    logging_strategy="steps",              # Log every `logging_steps`
    logging_steps=100,                     # Log every 100 steps

    # Evaluate, save checkpoints, and load the best model based on validation metrics
    evaluation_strategy="epoch" if val_dataset else "no", # Evaluate at the end of each epoch IF validation data exists
    save_strategy="epoch",                 # Save checkpoint every epoch
    load_best_model_at_end=True if val_dataset else False, # Load the model with the best eval metric at the end
    metric_for_best_model="accuracy" if val_dataset else None, # Use validation accuracy to determine the best model
    greater_is_better=True if val_dataset else None, # For accuracy, greater is better

    # Optional: Gradient Accumulation (useful if per_device_train_batch_size needs to be smaller)
    # gradient_accumulation_steps=1,       # effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

    report_to="none", # Can be "tensorboard", "wandb", etc. (configure if desired)
)

print("Training arguments configured.")

# --- 6. Initialize Trainer ---

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor, # Pass the processor as tokenizer
    data_collator=collate_fn, # Use our custom collate function
    compute_metrics=compute_metrics if val_dataset else None,
)

print("Trainer initialized.")

# --- 7. Train the Model ---

print("Starting training...")
trainer.train()
print("Training finished.")

# --- 8. Evaluate (Optional) ---
if val_dataset:
    print("Evaluating the best model on the validation set...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

# --- 9. Save the final model ---
print(f"Saving the final (or best) model and processor to {OUTPUT_DIR}/final_model")
trainer.save_model(f"{OUTPUT_DIR}/final_model")
processor.save_pretrained(f"{OUTPUT_DIR}/final_model")

print("Fine-tuning complete. Model saved for inference.")
