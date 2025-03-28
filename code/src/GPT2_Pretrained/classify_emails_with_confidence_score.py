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
from PyPDF2 import PdfReader
from docx import Document
import io
import numpy as np

# Load configuration from config.json
try:
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    
    request_types = config["request_types"]
    extract_field_patterns = config["extract_fields"]
    paths = config["paths"]
    processing = config["processing"]
    gpt2_config = config["gpt2"]

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

# Set up logging
logging.basicConfig(
    filename=email_processing_log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load GPT-2 model and tokenizer with device specification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
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

# Function to extract attachment text
def extract_attachment_text(msg):
    attachment_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    payload = part.get_payload(decode=True)
                    if filename.lower().endswith('.pdf'):
                        try:
                            pdf_reader = PdfReader(io.BytesIO(payload))
                            for page in pdf_reader.pages:
                                attachment_text += page.extract_text() or ""
                        except Exception as e:
                            logging.error(f"Error reading PDF attachment {filename}: {e}")
                    elif filename.lower().endswith('.docx'):
                        try:
                            doc = Document(io.BytesIO(payload))
                            for para in doc.paragraphs:
                                attachment_text += para.text + "\n"
                        except Exception as e:
                            logging.error(f"Error reading DOCX attachment {filename}: {e}")
    return attachment_text

# Modified to include context in email chain details
def extract_email_chain_details(msg):
    if msg is None:
        return [], [], [], "", []

    layers = []
    metadata = []
    attachments = []
    attachment_text = ""
    contexts = []  # Store context for each layer

    current_msg = msg
    layer = 0

    while current_msg:
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

        meta = {
            "From": current_msg.get("From", ""),
            "To": current_msg.get("To", ""),
            "Subject": current_msg.get("Subject", ""),
            "Date": current_msg.get("Date", "")
        }

        context = layers[-1] if layers else ""
        contexts.append(context)

        if layer == 0:
            attachment_text = extract_attachment_text(current_msg)
            layer_attachments = []
            if current_msg.is_multipart():
                for part in current_msg.walk():
                    if part.get_content_disposition() == "attachment":
                        filename = part.get_filename()
                        if filename:
                            payload = part.get_payload(decode=True)
                            attachment_hash = hashlib.md5(payload).hexdigest()
                            layer_attachments.append({
                                "filename": filename,
                                "hash": attachment_hash
                            })
            attachments.append(layer_attachments)
        else:
            attachments.append([])

        layers.append(text)
        metadata.append(meta)

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

    return layers, metadata, attachments, attachment_text, contexts

# Function to compute text similarity
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

# Modified classify_email to handle empty/invalid confidence values
def classify_email(text, context=""):
    logging.info("Classifying email content with context...")
    if not text:
        logging.warning("Empty email text, returning default classification.")
        return "Unknown", "Unknown", 0.0, 0.0

    prompt = f"""Given the previous email context:
{context}

Classify the following email body and identify the request type and sub-request type with individual confidence scores:
\n\n{text}\n\nRequest Type: 
Request Type Confidence: 
Sub-Request Type: 
Sub-Request Type Confidence: """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=gpt2_max_length_input, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=gpt2_max_length_output,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Log the raw response for debugging
        logging.debug(f"GPT-2 response: {response}")

        # Extract logits for confidence calculation
        logits = torch.stack(outputs.scores, dim=1)
        probs = torch.softmax(logits, dim=-1)
        avg_confidence = float(probs.max().item())  # Fallback confidence

    except Exception as e:
        logging.error(f"Error during GPT-2 classification: {e}")
        return "Unknown", "Unknown", 0.0, 0.0

    request_type_match = re.search(r"Request Type: (.*?)\n", response)
    request_conf_match = re.search(r"Request Type Confidence: (.*?)\n", response)
    sub_request_type_match = re.search(r"Sub-Request Type: (.*?)\n", response)
    sub_request_conf_match = re.search(r"Sub-Request Type Confidence: (.*?)$", response, re.MULTILINE)

    request_type = request_type_match.group(1) if request_type_match else "Unknown"
    sub_request_type = sub_request_type_match.group(1) if sub_request_type_match else "Unknown"
    
    # Safely handle request_confidence
    request_confidence = avg_confidence
    if request_conf_match:
        conf_str = request_conf_match.group(1).strip()
        try:
            request_confidence = float(conf_str) if conf_str else avg_confidence
        except ValueError:
            logging.warning(f"Invalid Request Type Confidence: '{conf_str}', using fallback: {avg_confidence}")
            request_confidence = avg_confidence

    # Safely handle sub_request_confidence
    sub_request_confidence = avg_confidence
    if sub_request_conf_match:
        conf_str = sub_request_conf_match.group(1).strip()
        try:
            sub_request_confidence = float(conf_str) if conf_str else avg_confidence
        except ValueError:
            logging.warning(f"Invalid Sub-Request Type Confidence: '{conf_str}', using fallback: {avg_confidence}")
            sub_request_confidence = avg_confidence

    logging.info(f"Classification - Request Type: {request_type} ({request_confidence}), Sub-Request Type: {sub_request_type} ({sub_request_confidence})")
    return request_type, sub_request_type, request_confidence, sub_request_confidence

# Modified extract_fields to include confidence
def extract_fields(attachment_text):
    logging.info("Extracting fields from attachment...")
    fields = {}
    confidences = {}
    
    for field_name, pattern in extract_field_patterns.items():
        match = re.search(pattern, attachment_text)
        if match:
            value = match.group(1)
            confidence = min(1.0, len(match.group(0)) / len(attachment_text) * 5)
            if field_name == "deal_amount":
                fields[field_name] = f"${value}"
            else:
                fields[field_name] = value
            confidences[field_name] = confidence
        else:
            fields[field_name] = "Unknown"
            confidences[field_name] = 0.0
    
    overall_confidence = np.mean(list(confidences.values())) if confidences else 0.0
    logging.info(f"Extracted fields: {fields}, Confidences: {confidences}")
    return fields, confidences, overall_confidence

# Modified detect_primary_intent to include confidence
def detect_primary_intent(text):
    logging.info("Detecting primary intent...")
    max_similarity = 0.0
    detected_intent = "Unknown"
    
    for request_type in request_types.keys():
        similarity = compute_text_similarity(text.lower(), request_type.lower())
        if similarity > max_similarity:
            max_similarity = similarity
            detected_intent = request_type
    
    confidence = max_similarity if detected_intent != "Unknown" else 0.0
    logging.info(f"Primary intent: {detected_intent}, Confidence: {confidence}")
    return detected_intent, confidence

# Function to detect duplicates (unchanged)
def detect_duplicates(email_chains, similarity_threshold=0.95):
    logging.info("Detecting duplicate email chains...")
    duplicates = []
    chain_details = []

    for email_id, layers, metadata, attachments, _ in email_chains:
        chain_signature = {
            "layers": layers,
            "metadata": metadata,
            "attachments": attachments
        }
        chain_details.append((email_id, chain_signature))

    for i, (email_id_i, chain_i) in enumerate(chain_details):
        for j, (email_id_j, chain_j) in enumerate(chain_details[i+1:], start=i+1):
            if len(chain_i["layers"]) != len(chain_j["layers"]):
                continue

            is_duplicate = True
            layer_similarities = []
            for layer_idx in range(len(chain_i["layers"])):
                text_sim = compute_text_similarity(chain_i["layers"][layer_idx], chain_j["layers"][layer_idx])
                layer_similarities.append(text_sim)
                if text_sim < similarity_threshold:
                    is_duplicate = False
                    break

                meta_i = chain_i["metadata"][layer_idx]
                meta_j = chain_j["metadata"][layer_idx]
                if (meta_i["From"] != meta_j["From"] or
                    meta_i["To"] != meta_j["To"] or
                    meta_i["Subject"] != meta_j["Subject"]):
                    is_duplicate = False
                    break

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

# Modified store_classification to include all confidence scores
def store_classification(email_id, request_type, sub_request_type, request_confidence, sub_request_confidence,
                       fields, field_confidences, fields_confidence, primary_intent, intent_confidence):
    safe_request_type = re.sub(r'[^a-zA-Z0-9\-]', '_', request_type)
    folder_path = f"{classifications_dir}/{safe_request_type}/email_{email_id}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Calculate overall confidence (weighted average)
    overall_confidence = np.mean([
        request_confidence * 0.25,
        sub_request_confidence * 0.25,
        fields_confidence * 0.3,
        intent_confidence * 0.2
    ])
    
    result = {
        "request_type": request_type,
        "request_type_confidence": request_confidence,
        "sub_request_type": sub_request_type,
        "sub_request_type_confidence": sub_request_confidence,
        "fields": fields,
        "field_confidences": field_confidences,
        "primary_intent": primary_intent,
        "intent_confidence": intent_confidence,
        "confidence": {
            "classification": {
                "request_type": request_confidence,
                "sub_request_type": sub_request_confidence
            },
            "fields": fields_confidence,
            "intent": intent_confidence,
            "overall": overall_confidence
        }
    }
    
    with open(f"{folder_path}/classification.json", "w") as f:
        json.dump(result, f, indent=4)
    
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
    layers, metadata, attachments, attachment_text, contexts = extract_email_chain_details(msg)
    
    text = layers[0] if layers else ""
    context = contexts[0] if contexts else ""
    email_chains.append((i, layers, metadata, attachments, attachment_text))

    # Classify with context
    request_type, sub_request_type, request_confidence, sub_request_confidence = classify_email(text, context)

    # Extract fields with confidence
    fields, field_confidences, fields_confidence = extract_fields(attachment_text) if attachment_text else ({}, {}, 0.0)

    # Detect primary intent with confidence
    primary_intent, intent_confidence = detect_primary_intent(text)

    # Store classification with all confidence scores
    store_classification(
        i, request_type, sub_request_type, request_confidence, sub_request_confidence,
        fields, field_confidences, fields_confidence, primary_intent, intent_confidence
    )

# Detect duplicates
duplicates = detect_duplicates(email_chains, similarity_threshold=0.95)
logging.info(f"Final duplicate email chains: {duplicates}")

# Store duplicate information
os.makedirs(classifications_dir, exist_ok=True)
with open(f"{classifications_dir}/{duplicates_file}", "w") as f:
    json.dump({"duplicate_email_chains": duplicates}, f, indent=4)

print(f"Processed {num_emails} emails. Check {email_processing_log_file} for details and {classifications_dir}/ for results.")