import os
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile

model_name = 'OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract'
local_dir = '../container/model/openalex-bert-model'

# Function to check if model is already downloaded
def is_model_updated(local_dir):
    # Check if the model and tokenizer directories already exist and are not empty
    return os.path.exists(os.path.join(local_dir, 'pytorch_model.bin')) and \
           os.path.exists(os.path.join(local_dir, 'tokenizer_config.json'))

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# If the model is already downloaded, skip the download
if not is_model_updated(local_dir):
    print("Downloading and saving model and tokenizer...")

    # Use a temporary directory to download and update the model
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Save to temporary directory
            tokenizer.save_pretrained(temp_dir)
            model.save_pretrained(temp_dir)

            # If everything is successful, update the original directory
            if is_model_updated(temp_dir):
                shutil.rmtree(local_dir, ignore_errors=True)  # Remove the old model directory if needed
                shutil.move(temp_dir, local_dir)  # Move the new model and tokenizer to the permanent directory

            print(f"Model and tokenizer saved to {local_dir}")

    except Exception as e:
        print(f"Failed to download or save the model: {e}")
        print("Old model files are preserved. No update was made.")
else:
    print(f"Model and tokenizer are already up to date in {local_dir}")
