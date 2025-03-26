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
from PyPDF2 import PdfReader
from docx import Document
import io
from typing import Tuple, List, Dict, Any, Optional

# Load configuration
try:
    with open("config_advanced.json", "r") as config_file:
        config = json.load(config_file)
    
    request_types_config = config["request_types"]
    paths = config["paths"]
    processing = config["processing"]
    email_types = config["email_types"]
    attachment_fields = config["attachment_fields"]

    bert_config = config.get("bert", {"max_length_input": 512})
    bert_max_length = bert_config["max_length_input"]

    email_files_dir = paths["email_files_dir"]
    classifications_dir = paths["classifications_dir"]
    email_processing_log_file = paths["email_processing_log_file"]
    duplicates_file = paths["duplicates_file"]
    num_emails_per_type = processing["num_emails_per_type"]
    num_duplicate_emails = processing["num_duplicate_emails"]

except FileNotFoundError:
    logging.error("Configuration file 'config_advanced.json' not found.")
    raise
except json.JSONDecodeError as e:
    logging.error(f"Error decoding config_advanced.json: {e}")
    raise
except KeyError as e:
    logging.error(f"Missing required key in config_advanced.json: {e}")
    raise

# Set up logging
logging.basicConfig(
    filename=email_processing_log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define request types
REQUEST_TYPES = list(request_types_config.keys())
SUB_REQUEST_TYPES = {key: value for key, value in request_types_config.items()}
ALL_SUB_REQUEST_TYPES = list(dict.fromkeys([item for sub_list in SUB_REQUEST_TYPES.values() for item in sub_list]))

# Load BERT models
try:
    request_type_model_path = os.path.abspath("./request_type_model")
    sub_request_type_model_path = os.path.abspath("./sub_request_type_model")

    if not os.path.exists(request_type_model_path) or not os.path.exists(sub_request_type_model_path):
        raise FileNotFoundError("BERT model directory not found")

    tokenizer = BertTokenizer.from_pretrained(request_type_model_path, local_files_only=True)
    request_model = BertForSequenceClassification.from_pretrained(request_type_model_path, local_files_only=True)
    sub_request_model = BertForSequenceClassification.from_pretrained(sub_request_type_model_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    request_model.to(device)
    sub_request_model.to(device)
    request_model.eval()
    sub_request_model.eval()
    logging.info(f"Loaded BERT models on device: {device}")
except Exception as e:
    logging.error(f"Error loading BERT models: {e}")
    raise

def parse_eml_file(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, 'rb') as f:
            return BytesParser(policy=policy.default).parse(f)
    except Exception as e:
        logging.error(f"Error parsing .eml file {file_path}: {e}")
        return None

def extract_attachment_text(msg: Any) -> str:
    attachment_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    payload = part.get_payload(decode=True)
                    logging.info(f"Extracting text from attachment: {filename}")
                    if filename.lower().endswith('.pdf'):
                        try:
                            pdf_reader = PdfReader(io.BytesIO(payload))
                            attachment_text += "".join(page.extract_text() or "" for page in pdf_reader.pages) + "\n"
                        except Exception as e:
                            logging.error(f"Error reading PDF {filename}: {e}")
                    elif filename.lower().endswith('.docx'):
                        try:
                            doc = Document(io.BytesIO(payload))
                            attachment_text += "".join(para.text for para in doc.paragraphs) + "\n"
                        except Exception as e:
                            logging.error(f"Error reading DOCX {filename}: {e}")
    return attachment_text.strip()

def extract_email_chain_details(msg: Any) -> Tuple[List[str], List[Dict[str, str]], List[List[Dict[str, str]]], str]:
    if msg is None:
        return [], [], [], ""

    layers, metadata, attachments = [], [], []
    current_msg = msg
    attachment_text = ""

    while current_msg:
        text = (next((part.get_payload(decode=True).decode('utf-8', errors='ignore') 
                     for part in current_msg.walk() if part.get_content_type() == "text/plain"), "")
                if current_msg.is_multipart() else 
                current_msg.get_payload(decode=True).decode('utf-8', errors='ignore'))

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
                        layer_attachments.append({
                            "filename": filename,
                            "hash": hashlib.md5(payload).hexdigest()
                        })
                        if not attachment_text:
                            attachment_text = extract_attachment_text(current_msg)
        attachments.append(layer_attachments)

        layers.append(text.strip())
        metadata.append(meta)
        current_msg = next((part.get_payload()[0] for part in current_msg.walk() 
                          if part.get_content_type() == "message/rfc822"), None)

    return layers, metadata, attachments, attachment_text

def compute_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def classify_email(text: str) -> Tuple[str, str, float, float]:
    if not text:
        return "Unknown", "Unknown", 0.0, 0.0

    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=bert_max_length, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = request_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_request_idx = torch.argmax(probabilities, dim=1).item()
            request_confidence = probabilities[0][predicted_request_idx].item()
            request_type = REQUEST_TYPES[predicted_request_idx]

        sub_request_type = "General"
        sub_request_confidence = 1.0
        if request_type in SUB_REQUEST_TYPES and SUB_REQUEST_TYPES[request_type]:
            sub_request_candidates = SUB_REQUEST_TYPES[request_type]
            if sub_request_candidates != [request_type]:
                with torch.no_grad():
                    outputs = sub_request_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_sub_request_idx = torch.argmax(probabilities, dim=1).item()
                    sub_request_confidence = probabilities[0][predicted_sub_request_idx].item()
                    global_sub_request = ALL_SUB_REQUEST_TYPES[predicted_sub_request_idx]
                    if global_sub_request in sub_request_candidates:
                        sub_request_type = global_sub_request
                    else:
                        text_lower = text.lower()
                        for sub_type in sub_request_candidates:
                            if sub_type.lower() in text_lower:
                                sub_request_type = sub_type
                                sub_request_confidence = 0.9
                                break
                        else:
                            sub_request_type = sub_request_candidates[0]
                            sub_request_confidence = 0.5

    except Exception as e:
        logging.error(f"Error during BERT classification: {e}")
        return "Unknown", "Unknown", 0.0, 0.0

    if request_type not in REQUEST_TYPES:
        text_lower = text.lower()
        for req_type in REQUEST_TYPES:
            if req_type.lower() in text_lower:
                request_type = req_type
                request_confidence = 0.9
                sub_request_candidates = SUB_REQUEST_TYPES.get(request_type, [])
                if sub_request_candidates == [request_type]:
                    sub_request_type = "General"
                    sub_request_confidence = 1.0
                else:
                    for sub_type in sub_request_candidates:
                        if sub_type.lower() in text_lower:
                            sub_request_type = sub_type
                            sub_request_confidence = 0.9
                            break
                    else:
                        sub_request_type = sub_request_candidates[0] if sub_request_candidates else "General"
                        sub_request_confidence = 0.5
                break
        else:
            return "Unknown", "Unknown", 0.0, 0.0

    logging.info(f"Classification - Request: {request_type} ({request_confidence:.4f}), Sub-Request: {sub_request_type} ({sub_request_confidence:.4f})")
    return request_type, sub_request_type, request_confidence, sub_request_confidence

def extract_fields(attachment_text: str) -> Dict[str, str]:
    fields = {}
    patterns = {
        "deal_name": r"Deal Name: (Deal-\d+)",
        "original_amount": r"Original Amount: \$ ([\d,]+\.\d{3})",
        "adjusted_amount": r"Adjusted Amount: \$ ([\d,]+\.\d{3})",
        "effective_date": r"Effective Date: (\d{2}-[A-Za-z]{3}-\d{4})",
        "expiration_date": r"Expiration Date: (\d{2}-[A-Za-z]{3}-\d{4})",
        "borrower": r"Borrower: (.+?)(?:\n|$)",
        "bank_name": r"Bank Name: (.+?)(?:\n|$)",
        "account_number": r"Account #: (\d+)",
        "account_name": r"Account Name: (.+?)(?:\n|$)"
    }
    
    request_matches = re.findall(r"Request Type: (.+?)\nDescription: (.+?)(?:\n|$)", attachment_text)
    fields["requests"] = [{"request_type": rt, "sub_request_type": srt} for rt, srt in request_matches]

    for field_name, pattern in patterns.items():
        match = re.search(pattern, attachment_text) if attachment_text else None
        fields[field_name] = match.group(1) if match else "Unknown"

    return fields

def detect_primary_intent(text: str, attachment_text: str = "") -> Tuple[str, float]:
    combined_text = text + "\n" + attachment_text
    if not combined_text.strip():
        return "Unknown", 0.0
    
    # Count occurrences of each request type
    intent_scores = {rt: combined_text.lower().count(rt.lower()) for rt in REQUEST_TYPES}
    total_counts = sum(intent_scores.values())
    
    if total_counts == 0:
        return "Unknown", 0.0
    
    # Calculate confidence as the proportion of the winning intent's mentions
    primary_intent = max(intent_scores, key=intent_scores.get)
    intent_confidence = intent_scores[primary_intent] / total_counts if total_counts > 0 else 0.0
    
    logging.info(f"Primary intent: {primary_intent} (Confidence: {intent_confidence:.4f})")
    return primary_intent, intent_confidence

def detect_duplicates(email_chains: List[Tuple[str, str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]], str]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
    duplicates = []
    chain_details = [(type_id, email_id, {"layers": layers, "metadata": metadata, "attachments": attachments}) 
                    for type_id, email_id, layers, metadata, attachments, _ in email_chains]

    for i, (type_id_i, email_id_i, chain_i) in enumerate(chain_details):
        for j, (type_id_j, email_id_j, chain_j) in enumerate(chain_details[i+1:], start=i+1):
            if len(chain_i["layers"]) != len(chain_j["layers"]):
                continue

            is_duplicate = True
            layer_similarities = []
            for layer_idx in range(len(chain_i["layers"])):
                text_sim = compute_text_similarity(chain_i["layers"][layer_idx], chain_j["layers"][layer_idx])
                layer_similarities.append(text_sim)
                if (text_sim < similarity_threshold or 
                    any(chain_i["metadata"][layer_idx][k] != chain_j["metadata"][layer_idx][k] for k in ["From", "To", "Subject"]) or
                    len(chain_i["attachments"][layer_idx]) != len(chain_j["attachments"][layer_idx]) or
                    any(att_i["hash"] != att_j["hash"] for att_i, att_j in zip(chain_i["attachments"][layer_idx], chain_j["attachments"][layer_idx]))):
                    is_duplicate = False
                    break

            if is_duplicate:
                duplicates.append({
                    "email_1": {"type_id": type_id_i, "email_id": email_id_i},
                    "email_2": {"type_id": type_id_j, "email_id": email_id_j},
                    "layer_similarities": layer_similarities
                })

    return duplicates

def store_classification(type_id: str, email_id: int, request_type: str, sub_request_type: str, request_confidence: float, sub_request_confidence: float, fields: Dict[str, str], primary_intent: str, primary_intent_confidence: float) -> None:
    if request_type not in REQUEST_TYPES:
        request_type, request_confidence = "Unknown", 0.0
    
    safe_request_type = re.sub(r'[^a-zA-Z0-9\-]', '_', request_type)
    folder_path = f"{classifications_dir}/type_{type_id}/{safe_request_type}/email_{email_id}"
    os.makedirs(folder_path, exist_ok=True)
    
    result = {
        "request_type": request_type,
        "request_confidence": round(request_confidence, 4),
        "sub_request_type": sub_request_type,
        "sub_request_confidence": round(sub_request_confidence, 4),
        "fields": fields,
        "primary_intent": primary_intent,
        "primary_intent_confidence": round(primary_intent_confidence, 4)
    }
    
    with open(f"{folder_path}/classification.json", "w") as f:
        json.dump(result, f, indent=4)
    
    eml_source = f"{email_files_dir}/type_{type_id}/email_{type_id}_{email_id}.eml"
    eml_dest = f"{folder_path}/email_{type_id}_{email_id}.eml"
    shutil.copy(eml_source, eml_dest)
    logging.info(f"Stored classification for type_{type_id}/email_{email_id}")

# Main processing
email_chains = []
total_emails_processed = 0

for email_type in email_types:
    type_id = email_type["type_id"]
    num_emails = email_type.get("num_emails", num_emails_per_type)
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

        msg = parse_eml_file(file_path)
        layers, metadata, attachments, attachment_text = extract_email_chain_details(msg)
        full_text = "\n".join(layers) if layers else ""
        email_chains.append((type_id, email_id, layers, metadata, attachments, attachment_text))

        request_type, sub_request_type, request_confidence, sub_request_confidence = classify_email(full_text)
        fields = extract_fields(attachment_text)
        primary_intent, primary_intent_confidence = detect_primary_intent(full_text, attachment_text)
        store_classification(type_id, i, request_type, sub_request_type, request_confidence, sub_request_confidence, fields, primary_intent, primary_intent_confidence)
        total_emails_processed += 1

duplicates = detect_duplicates(email_chains)
os.makedirs(classifications_dir, exist_ok=True)
with open(f"{classifications_dir}/{duplicates_file}", "w") as f:
    json.dump({"duplicate_email_chains": duplicates}, f, indent=4)

print(f"Processed {total_emails_processed} emails. Check {email_processing_log_file} for details and {classifications_dir}/ for results.")