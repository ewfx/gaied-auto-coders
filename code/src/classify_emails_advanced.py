import os
import re
import json
import logging
import shutil
from email import policy
from email.parser import BytesParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load configuration from config_advanced.json
try:
    with open("config_advanced.json", "r") as config_file:
        config = json.load(config_file)
    
    # Extract configuration sections
    request_types = config["request_types"]
    paths = config["paths"]
    processing = config["processing"]
    email_types = config["email_types"]
    gpt2_config = config["gpt2"]

    # Extract specific configuration values
    email_files_dir = paths["email_files_dir"]
    classifications_dir = paths["classifications_dir"]
    email_processing_log_file = paths["email_processing_log_file"]
    duplicates_file = paths["duplicates_file"]
    num_emails_per_type = processing["num_emails_per_type"]
    num_duplicate_emails = processing["num_duplicate_emails"]
    gpt2_model_name = gpt2_config["model_name"]
    gpt2_max_length_input = gpt2_config["max_length_input"]
    gpt2_max_length_output = gpt2_config["max_length_output"]

except FileNotFoundError:
    logging.error("Configuration file 'config_advanced.json' not found.")
    raise FileNotFoundError("Configuration file 'config_advanced.json' not found.")
except json.JSONDecodeError as e:
    logging.error(f"Error decoding config_advanced.json: {e}")
    raise json.JSONDecodeError(f"Error decoding config_advanced.json: {e}")
except KeyError as e:
    logging.error(f"Missing required key in config_advanced.json: {e}")
    raise KeyError(f"Missing required key in config_advanced.json: {e}")

# Set up logging for email processing
logging.basicConfig(
    filename=email_processing_log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load GPT-2 model and tokenizer
try:
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    model.eval()
except Exception as e:
    logging.error(f"Error loading GPT-2 model or tokenizer: {e}")
    raise Exception(f"Error loading GPT-2 model or tokenizer: {e}")

# Function to parse .eml file
def parse_eml_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        return msg
    except Exception as e:
        logging.error(f"Error parsing .eml file {file_path}: {e}")
        return None

# Function to extract text from email
def extract_text_from_email(msg):
    if msg is None:
        return ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_payload(decode=True).decode()
                except Exception as e:
                    logging.error(f"Error decoding email part: {e}")
                    return ""
    else:
        try:
            return msg.get_payload(decode=True).decode()
        except Exception as e:
            logging.error(f"Error decoding email payload: {e}")
            return ""
    return ""

# Function to classify email using GPT-2
def classify_email(text):
    logging.info("Classifying email content...")
    if not text:
        logging.warning("Empty email text, returning default classification.")
        return "Unknown", "Unknown"

    prompt = f"Classify the following email and identify the request type and sub-request type:\n\n{text}\n\nRequest Type: "
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=gpt2_max_length_input, truncation=True)
        outputs = model.generate(**inputs, max_length=gpt2_max_length_output, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during GPT-2 classification: {e}")
        return "Unknown", "Unknown"

    # Extract request type and sub-request type from response
    request_type_match = re.search(r"Request Type: (.*?)\n", response)
    sub_request_type_match = re.search(r"Sub-Request Type: (.*?)\n", response)

    request_type = request_type_match.group(1) if request_type_match else "Unknown"
    sub_request_type = sub_request_type_match.group(1) if sub_request_type_match else "Unknown"
    logging.info(f"Classification result - Request Type: {request_type}, Sub-Request Type: {sub_request_type}")
    return request_type, sub_request_type

# Function to extract fields based on patterns
def extract_fields(text):
    logging.info("Extracting fields from email...")
    fields = {}
    patterns = {
        "deal_id": r"Deal Name: Deal-(\d+)",
        "amount": r"Amount: \$(\d+)",
        "expiration_date": r"Expiration Date: (.*?)\n"
    }
    for field_name, pattern in patterns.items():
        match = re.search(pattern, text)
        fields[field_name] = match.group(1) if match else "Unknown"
    logging.info(f"Extracted fields: {fields}")
    return fields

# Function to detect primary intent in multi-request emails
def detect_primary_intent(text):
    logging.info("Detecting primary intent...")
    for request_type in request_types.keys():
        if request_type in text:
            logging.info(f"Primary intent detected: {request_type}")
            return request_type
    logging.info("Primary intent not detected, defaulting to Unknown")
    return "Unknown"

# Function to detect duplicates
def detect_duplicates(email_texts):
    logging.info("Detecting duplicate emails...")
    duplicates = []
    seen = set()
    for i, (type_id, email_id, text) in enumerate(email_texts):
        if text in seen:
            duplicates.append({"type_id": type_id, "email_id": email_id})
        else:
            seen.add(text)
    logging.info(f"Duplicate emails: {duplicates}")
    return duplicates

# Function to store classification results in individual folders
def store_classification(type_id, email_id, request_type, sub_request_type, fields, primary_intent):
    # Use the request_type to create a subfolder under classifications/type_{type_id}
    safe_request_type = re.sub(r'[^a-zA-Z0-9\-]', '_', request_type)
    folder_path = f"{classifications_dir}/type_{type_id}/{safe_request_type}/email_{email_id}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Store classification result in JSON
    result = {
        "request_type": request_type,
        "sub_request_type": sub_request_type,
        "fields": fields,
        "primary_intent": primary_intent
    }
    
    try:
        with open(f"{folder_path}/classification.json", "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing classification.json for type_{type_id}/email_{email_id}: {e}")
        return
    
    # Copy the corresponding .eml file to the folder
    eml_source_path = f"{email_files_dir}/type_{type_id}/email_type_{type_id}_{email_id}.eml"
    eml_dest_path = f"{folder_path}/email_type_{type_id}_{email_id}.eml"
    try:
        shutil.copy(eml_source_path, eml_dest_path)
        logging.info(f"Copied {eml_source_path} to {eml_dest_path}")
    except Exception as e:
        logging.error(f"Error copying .eml file for type_{type_id}/email_{email_id}: {e}")

    logging.info(f"Stored classification for type_{type_id}/email_{email_id} in {folder_path}")

# Process all emails
email_texts = []
total_emails_processed = 0

for email_type in email_types:
    type_id = email_type["type_id"]
    num_emails = email_type.get("num_emails", num_emails_per_type)  # Use num_emails if specified (e.g., for type 17)

    logging.info(f"Processing emails for type_{type_id}...")
    type_dir = f"{email_files_dir}/type_{type_id}"
    
    if not os.path.exists(type_dir):
        logging.warning(f"Directory {type_dir} does not exist, skipping...")
        continue

    for i in range(num_emails):
        email_id = f"{type_id}_{i}"
        file_path = f"{type_dir}/email_{email_id}.eml"
        
        if not os.path.exists(file_path):
            logging.warning(f"File {file_path} does not exist, skipping...")
            continue

        logging.info(f"Processing email_{email_id}.eml...")
        msg = parse_eml_file(file_path)
        text = extract_text_from_email(msg)
        email_texts.append((type_id, email_id, text))

        # Classify email
        request_type, sub_request_type = classify_email(text)

        # Extract fields
        fields = extract_fields(text)

        # Detect primary intent if multi-request
        primary_intent = detect_primary_intent(text)

        # Store classification
        store_classification(type_id, i, request_type, sub_request_type, fields, primary_intent)
        total_emails_processed += 1

# Detect duplicates across all emails
duplicates = detect_duplicates(email_texts)
logging.info(f"Final duplicate emails: {duplicates}")

# Store duplicate information
os.makedirs(classifications_dir, exist_ok=True)
try:
    with open(f"{classifications_dir}/{duplicates_file}", "w") as f:
        json.dump({"duplicate_emails": duplicates}, f, indent=4)
except Exception as e:
    logging.error(f"Error writing duplicates file: {e}")

print(f"Processed {total_emails_processed} emails. Check {email_processing_log_file} for details and {classifications_dir}/ for results.")