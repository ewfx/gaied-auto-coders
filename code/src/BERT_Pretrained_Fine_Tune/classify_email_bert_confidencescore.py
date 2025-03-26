import os
import re
import json
import logging
import shutil
from email import policy
from email.parser import BytesParser
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from typing import Tuple, List, Dict, Any, Optional

# Load configuration from config_advanced.json
try:
    with open("config_advanced.json", "r") as config_file:
        config = json.load(config_file)
    
    request_types_config = config["request_types"]
    paths = config["paths"]
    processing = config["processing"]
    email_types = config["email_types"]

    bert_config = config.get("bert", {})
    bert_max_length = bert_config.get("max_length_input", 512)

    email_files_dir = paths["email_files_dir"]
    classifications_dir = paths["classifications_dir"]
    email_processing_log_file = paths["email_processing_log_file"]
    duplicates_file = paths["duplicates_file"]
    num_emails_per_type = processing["num_emails_per_type"]
    num_duplicate_emails = processing["num_duplicate_emails"]

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

# Define request types and sub-request types from the config
REQUEST_TYPES = list(request_types_config.keys())
SUB_REQUEST_TYPES = {key: value for key, value in request_types_config.items()}
ALL_SUB_REQUEST_TYPES = list(dict.fromkeys([item for sub_list in SUB_REQUEST_TYPES.values() for item in sub_list]))

# Load fine-tuned BERT models and tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("./request_type_model")
    request_model = BertForSequenceClassification.from_pretrained("./request_type_model")
    sub_request_model = BertForSequenceClassification.from_pretrained("./sub_request_type_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    request_model.to(device)
    sub_request_model.to(device)
    request_model.eval()
    sub_request_model.eval()
    logging.info(f"Loaded fine-tuned BERT models on device: {device}")
except Exception as e:
    logging.error(f"Error loading fine-tuned BERT models or tokenizer: {e}")
    raise Exception(f"Error loading fine-tuned BERT models or tokenizer: {e}")

# Function to parse .eml file
def parse_eml_file(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        return msg
    except Exception as e:
        logging.error(f"Error parsing .eml file {file_path}: {e}")
        return None

# Function to extract text, metadata, and attachments from each layer of the email chain
def extract_email_chain_details(msg: Any) -> Tuple[List[str], List[Dict[str, str]], List[List[Dict[str, str]]]]:
    if msg is None:
        return [], [], []

    layers = []
    metadata = []
    attachments = []

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

        layers.append(text)
        metadata.append(meta)
        attachments.append(layer_attachments)

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
def compute_text_similarity(text1: str, text2: str) -> float:
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

# Function to extract sub-request type from email text as a fallback
def extract_sub_request_from_text(text: str, sub_request_candidates: List[str]) -> Tuple[Optional[str], float]:
    logging.info("Extracting sub-request type from email text as fallback...")
    pattern = r"Sub-Request Type: (.*?)(?:\n|$)"
    match = re.search(pattern, text)
    if match:
        extracted_sub_request = match.group(1).strip()
        if extracted_sub_request in sub_request_candidates:
            logging.info(f"Extracted sub-request type from text: {extracted_sub_request}")
            return extracted_sub_request, 0.9  # High confidence for explicit match
        else:
            logging.warning(f"Extracted sub-request type '{extracted_sub_request}' not in valid candidates: {sub_request_candidates}")
    return None, 0.0  # No match found

# Function to classify email using fine-tuned BERT models with confidence scores
def classify_email(text: str) -> Tuple[str, float, Optional[str], float]:
    logging.info("Classifying email content with fine-tuned BERT models...")
    logging.info(f"Email text for classification: {text}")
    if not text:
        logging.warning("Empty email text, defaulting to first request type with no sub-request.")
        return REQUEST_TYPES[0], 0.5, None, 0.0  # Low confidence for default

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=bert_max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Step 1: Classify Request Type
        with torch.no_grad():
            outputs = request_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_request_idx = torch.argmax(probabilities, dim=1).item()
            request_confidence = probabilities[0][predicted_request_idx].item()
            request_type = REQUEST_TYPES[predicted_request_idx]

        # Step 2: Classify Sub-Request Type
        sub_request_type = None
        sub_request_confidence = 0.0
        if request_type in SUB_REQUEST_TYPES and SUB_REQUEST_TYPES[request_type]:
            sub_request_candidates = SUB_REQUEST_TYPES[request_type]
            if sub_request_candidates == [request_type]:
                sub_request_type = None
                sub_request_confidence = 0.0  # No sub-request applicable
            else:
                with torch.no_grad():
                    outputs = sub_request_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_sub_request_idx = torch.argmax(probabilities, dim=1).item()
                    sub_request_confidence = probabilities[0][predicted_sub_request_idx].item()
                    global_sub_request = ALL_SUB_REQUEST_TYPES[predicted_sub_request_idx]
                    logging.info(f"Predicted sub-request type: {global_sub_request}, Valid candidates: {sub_request_candidates}")
                    if global_sub_request in sub_request_candidates:
                        sub_request_type = global_sub_request
                    else:
                        logging.warning(f"Predicted sub-request type '{global_sub_request}' not valid for request type '{request_type}'. Falling back to text extraction.")
                        sub_request_type, sub_request_confidence = extract_sub_request_from_text(text, sub_request_candidates)
                        if sub_request_type is None:
                            sub_request_type = sub_request_candidates[0]
                            sub_request_confidence = 0.5  # Moderate confidence for default

        logging.info(f"Classification result - Request Type: {request_type} ({request_confidence:.4f}), Sub-Request Type: {sub_request_type if sub_request_type else 'None'} ({sub_request_confidence:.4f})")
        return request_type, request_confidence, sub_request_type, sub_request_confidence

    except Exception as e:
        logging.error(f"Error during BERT classification: {e}")
        return REQUEST_TYPES[0], 0.5, None, 0.0  # Default with low confidence

# Function to extract fields based on patterns
def extract_fields(text: str) -> Dict[str, str]:
    logging.info("Extracting fields from email...")
    fields = {}
    patterns = {
        "deal_id": r"Deal Name: Deal-(\d+)",
        "amount": r"Amount: \$(\d+)",
        "expiration_date": r"Expiration Date: (.*?)\n"
    }
    for field_name, pattern in patterns.items():
        match = re.search(pattern, text)
        fields[field_name] = match.group(1) if match else "Not Found"
    logging.info(f"Extracted fields: {fields}")
    return fields

# Function to detect primary intent with confidence score
def detect_primary_intent(text: str) -> Tuple[str, float]:
    logging.info("Detecting primary intent...")
    if not text.strip():
        logging.info("Empty text, defaulting to first request type with zero confidence")
        return REQUEST_TYPES[0], 0.0

    intent_scores = {rt: text.upper().count(rt.upper()) for rt in REQUEST_TYPES}
    total_counts = sum(intent_scores.values())
    
    if total_counts == 0:
        logging.info("No request types found, defaulting to first request type with zero confidence")
        return REQUEST_TYPES[0], 0.0
    
    primary_intent = max(intent_scores, key=intent_scores.get)
    intent_confidence = intent_scores[primary_intent] / total_counts
    logging.info(f"Primary intent detected: {primary_intent} (Confidence: {intent_confidence:.4f})")
    return primary_intent, intent_confidence

# Function to detect duplicates by comparing email chains
def detect_duplicates(email_chains: List[Tuple[str, str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
    logging.info("Detecting duplicate email chains...")
    duplicates = []
    chain_details = []

    for type_id, email_id, layers, metadata, attachments in email_chains:
        chain_signature = {
            "layers": layers,
            "metadata": metadata,
            "attachments": attachments
        }
        chain_details.append((type_id, email_id, chain_signature))

    for i, (type_id_i, email_id_i, chain_i) in enumerate(chain_details):
        for j, (type_id_j, email_id_j, chain_j) in enumerate(chain_details[i+1:], start=i+1):
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
                    "email_1": {"type_id": type_id_i, "email_id": email_id_i},
                    "email_2": {"type_id": type_id_j, "email_id": email_id_j},
                    "layer_similarities": layer_similarities
                })
                logging.info(f"Found duplicate: type_{type_id_i}/email_{email_id_i} and type_{type_id_j}/email_{email_id_j}")

    logging.info(f"Duplicate email chains: {duplicates}")
    return duplicates

# Function to store classification results with confidence scores
def store_classification(type_id: str, email_id: int, request_type: str, request_confidence: float, 
                        sub_request_type: Optional[str], sub_request_confidence: float, 
                        fields: Dict[str, str], primary_intent: str, primary_intent_confidence: float) -> None:
    safe_request_type = re.sub(r'[^a-zA-Z0-9\-]', '_', request_type)
    folder_path = f"{classifications_dir}/type_{type_id}/{safe_request_type}/email_{email_id}"
    os.makedirs(folder_path, exist_ok=True)
    
    result = {
        "request_type": request_type,
        "request_confidence": round(request_confidence, 4),
        "sub_request_type": sub_request_type if sub_request_type else "None",
        "sub_request_confidence": round(sub_request_confidence, 4),
        "fields": fields,
        "primary_intent": primary_intent,
        "primary_intent_confidence": round(primary_intent_confidence, 4)
    }
    
    try:
        with open(f"{folder_path}/classification.json", "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing classification.json for type_{type_id}/email_{email_id}: {e}")
        return
    
    eml_source_path = f"{email_files_dir}/type_{type_id}/email_type_{type_id}_{email_id}.eml"
    eml_dest_path = f"{folder_path}/email_type_{type_id}_{email_id}.eml"
    try:
        shutil.copy(eml_source_path, eml_dest_path)
        logging.info(f"Copied {eml_source_path} to {eml_dest_path}")
    except Exception as e:
        logging.error(f"Error copying .eml file for type_{type_id}/email_{email_id}: {e}")

    logging.info(f"Stored classification for type_{type_id}/email_{email_id} in {folder_path}")

# Process all emails
email_chains = []
total_emails_processed = 0

for email_type in email_types:
    type_id = email_type["type_id"]
    num_emails = email_type.get("num_emails", num_emails_per_type)

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
        layers, metadata, attachments = extract_email_chain_details(msg)
        
        text = layers[0] if layers else ""
        email_chains.append((type_id, email_id, layers, metadata, attachments))

        request_type, request_confidence, sub_request_type, sub_request_confidence = classify_email(text)
        fields = extract_fields(text)
        primary_intent, primary_intent_confidence = detect_primary_intent(text)
        store_classification(type_id, i, request_type, request_confidence, sub_request_type, 
                            sub_request_confidence, fields, primary_intent, primary_intent_confidence)
        total_emails_processed += 1

# Detect duplicates across all email chains
duplicates = detect_duplicates(email_chains, similarity_threshold=0.95)
logging.info(f"Final duplicate email chains: {duplicates}")

# Store duplicate information
os.makedirs(classifications_dir, exist_ok=True)
try:
    with open(f"{classifications_dir}/{duplicates_file}", "w") as f:
        json.dump({"duplicate_email_chains": duplicates}, f, indent=4)
except Exception as e:
    logging.error(f"Error writing duplicates file: {e}")

print(f"Processed {total_emails_processed} emails. Check {email_processing_log_file} for details and {classifications_dir}/ for results.")