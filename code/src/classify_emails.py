import os
import re
import json
import logging
import shutil
from email import policy
from email.parser import BytesParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Load configuration from config.json
try:
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    
    # Extract configuration sections
    request_types = config["request_types"]
    extract_field_patterns = config["extract_fields"]
    paths = config["paths"]
    processing = config["processing"]
    gpt2_config = config["gpt2"]

    # Extract specific configuration values
    email_files_dir = paths["email_files_dir"]
    classifications_dir = paths["classifications_dir"]
    email_processing_log_file = paths["email_processing_log_file"]
    duplicates_file = paths["duplicates_file"]
    num_emails = processing["num_emails"]
    gpt2_model_name = gpt2_config["model_name"]
    gpt2_max_length_input = gpt2_config["max_length_input"]
    gpt2_max_length_output = gpt2_config["max_length_output"]

except FileNotFoundError:
    logging.error("Configuration file 'config.json' not found.")
    raise FileNotFoundError("Configuration file 'config.json' not found.")
except json.JSONDecodeError as e:
    logging.error(f"Error decoding config.json: {e}")
    raise json.JSONDecodeError(f"Error decoding config.json: {e}")
except KeyError as e:
    logging.error(f"Missing required key in config.json: {e}")
    raise KeyError(f"Missing required key in config.json: {e}")

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

# Function to extract text, metadata, and attachments from each layer of the email chain
def extract_email_chain_details(msg):
    if msg is None:
        return [], [], []

    layers = []
    metadata = []
    attachments = []

    current_msg = msg
    layer = 0

    while current_msg:
        # Extract text
        if current_msg.is_multipart():
            text = ""
            for part in current_msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        text = part.get_payload(decode=True).decode()
                        break
                    except Exception as e:
                        logging.error(f"Error decoding email part in layer {layer}: {e}")
                        text = ""
        else:
            try:
                text = current_msg.get_payload(decode=True).decode()
            except Exception as e:
                logging.error(f"Error decoding email payload in layer {layer}: {e}")
                text = ""

        # Extract metadata
        meta = {
            "From": current_msg.get("From", ""),
            "To": current_msg.get("To", ""),
            "Subject": current_msg.get("Subject", ""),
            "Date": current_msg.get("Date", "")
        }

        # Extract attachments
        layer_attachments = []
        if current_msg.is_multipart():
            for part in current_msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        payload = part.get_payload(decode=True)
                        # Compute a hash of the attachment content for comparison
                        attachment_hash = hashlib.md5(payload).hexdigest()
                        layer_attachments.append({
                            "filename": filename,
                            "hash": attachment_hash
                        })

        layers.append(text)
        metadata.append(meta)
        attachments.append(layer_attachments)

        # Move to the next layer (reply/forward)
        next_layer_text = ""
        for part in current_msg.walk():
            if part.get_content_type() == "message/rfc822":
                try:
                    next_msg = part.get_payload()[0]
                    current_msg = next_msg
                    layer += 1
                    break
                except Exception as e:
                    logging.error(f"Error parsing nested message in layer {layer}: {e}")
                    current_msg = None
                    break
            else:
                current_msg = None

    return layers, metadata, attachments

# Function to compute text similarity using cosine similarity
def compute_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        logging.error(f"Error computing text similarity: {e}")
        return 0.0

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

# Function to extract fields based on config
def extract_fields(text):
    logging.info("Extracting fields from email...")
    fields = {}
    for field_name, pattern in extract_field_patterns.items():
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

# Function to detect duplicates by comparing email chains
def detect_duplicates(email_chains, similarity_threshold=0.95):
    logging.info("Detecting duplicate email chains...")
    duplicates = []
    chain_details = []

    # Store details of each email chain for comparison
    for email_id, layers, metadata, attachments in email_chains:
        chain_signature = {
            "layers": layers,
            "metadata": metadata,
            "attachments": attachments
        }
        chain_details.append((email_id, chain_signature))

    # Compare each pair of email chains
    for i, (email_id_i, chain_i) in enumerate(chain_details):
        for j, (email_id_j, chain_j) in enumerate(chain_details[i+1:], start=i+1):
            # Check if the number of layers matches
            if len(chain_i["layers"]) != len(chain_j["layers"]):
                continue

            # Compare each layer
            is_duplicate = True
            layer_similarities = []
            for layer_idx in range(len(chain_i["layers"])):
                # Compare text similarity
                text_sim = compute_text_similarity(chain_i["layers"][layer_idx], chain_j["layers"][layer_idx])
                layer_similarities.append(text_sim)
                if text_sim < similarity_threshold:
                    is_duplicate = False
                    break

                # Compare metadata (allow for small differences in Date)
                meta_i = chain_i["metadata"][layer_idx]
                meta_j = chain_j["metadata"][layer_idx]
                if (meta_i["From"] != meta_j["From"] or
                    meta_i["To"] != meta_j["To"] or
                    meta_i["Subject"] != meta_j["Subject"]):
                    is_duplicate = False
                    break

                # Compare attachments
                attachments_i = chain_i["attachments"][layer_idx]
                attachments_j = chain_j["attachments"][layer_idx]
                if len(attachments_i) != len(attachments_j):
                    is_duplicate = False
                    break
                for att_i, att_j in zip(attachments_i, attachments_j):
                    if att_i["filename"] != att_j["filename"] or att_i["hash"] != att_j["hash"]:
                        is_duplicate = False
                        break
                if not is_duplicate:
                    break

            if is_duplicate:
                duplicates.append({
                    "email_1": email_id_i,
                    "email_2": email_id_j,
                    "layer_similarities": layer_similarities
                })
                logging.info(f"Found duplicate: email_{email_id_i} and email_{email_id_j}")

    logging.info(f"Duplicate email chains: {duplicates}")
    return duplicates

# Function to store classification results in individual folders
def store_classification(email_id, request_type, sub_request_type, fields, primary_intent):
    # Use the request_type to create a subfolder under classifications
    safe_request_type = re.sub(r'[^a-zA-Z0-9\-]', '_', request_type)
    folder_path = f"{classifications_dir}/{safe_request_type}/email_{email_id}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Store classification result in JSON
    result = {
        "request_type": request_type,
        "sub_request_type": sub_request_type,
        "fields": fields,
        "primary_intent": primary_intent
    }
    
    with open(f"{folder_path}/classification.json", "w") as f:
        json.dump(result, f, indent=4)
    
    # Copy the corresponding .eml file to the folder
    eml_source_path = f"{email_files_dir}/email_{email_id}.eml"
    eml_dest_path = f"{folder_path}/email_{email_id}.eml"
    try:
        shutil.copy(eml_source_path, eml_dest_path)
        logging.info(f"Copied {eml_source_path} to {eml_dest_path}")
    except Exception as e:
        logging.error(f"Error copying .eml file for email_{email_id}: {e}")

    logging.info(f"Stored classification for email_{email_id} in {folder_path}")

# Process all emails
email_chains = []
for i in range(num_emails):
    logging.info(f"Processing email_{i}.eml...")
    msg = parse_eml_file(f"{email_files_dir}/email_{i}.eml")
    layers, metadata, attachments = extract_email_chain_details(msg)
    
    # Use the first layer for classification (initial email)
    text = layers[0] if layers else ""
    email_chains.append((i, layers, metadata, attachments))

    # Classify email
    request_type, sub_request_type = classify_email(text)

    # Extract fields
    fields = extract_fields(text)

    # Detect primary intent if multi-request
    primary_intent = detect_primary_intent(text)

    # Store classification
    store_classification(i, request_type, sub_request_type, fields, primary_intent)

# Detect duplicates
duplicates = detect_duplicates(email_chains, similarity_threshold=0.95)
logging.info(f"Final duplicate email chains: {duplicates}")

# Store duplicate information
os.makedirs(classifications_dir, exist_ok=True)
with open(f"{classifications_dir}/{duplicates_file}", "w") as f:
    json.dump({"duplicate_email_chains": duplicates}, f, indent=4)

print(f"Processed {num_emails} emails. Check {email_processing_log_file} for details and {classifications_dir}/ for results.")