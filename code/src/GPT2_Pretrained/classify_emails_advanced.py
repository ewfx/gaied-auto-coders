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

# Load configuration from config_advanced.json
try:
    with open("config_advanced.json", "r") as config_file:
        config = json.load(config_file)
    
    request_types = config["request_types"]
    paths = config["paths"]
    processing = config["processing"]
    email_types = config["email_types"]
    attachment_fields = config["attachment_fields"]
    
    gpt2_config = config.get("gpt2", {
        "model_name": "gpt2",
        "max_length_input": 512,
        "max_length_output": 1024
    })

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

# Set up logging
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

# Function to extract attachment text
def extract_attachment_text(msg):
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
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    attachment_text += text + "\n"
                        except Exception as e:
                            logging.error(f"Error reading PDF attachment {filename}: {e}")
                    elif filename.lower().endswith('.docx'):
                        try:
                            doc = Document(io.BytesIO(payload))
                            for para in doc.paragraphs:
                                attachment_text += para.text + "\n"
                        except Exception as e:
                            logging.error(f"Error reading DOCX attachment {filename}: {e}")
    logging.info(f"Extracted attachment text: '{attachment_text}'")
    return attachment_text.strip()

# Function to extract email chain details
def extract_email_chain_details(msg):
    if msg is None:
        return [], [], [], ""

    layers = []
    metadata = []
    attachments = []
    attachment_text = ""

    current_msg = msg
    layer = 0

    while current_msg:
        if current_msg.is_multipart():
            text = ""
            for part in current_msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except Exception as e:
                        logging.error(f"Error decoding email part in layer {layer}: {e}")
                        text = ""
        else:
            try:
                text = current_msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except Exception as e:
                logging.error(f"Error decoding email payload in layer {layer}: {e}")
                text = ""

        meta = {
            "From": current_msg.get("From", ""),
            "To": current_msg.get("To", ""),
            "Subject": current_msg.get("Subject", ""),
            "Date": current_msg.get("Date", "")
        }

        if current_msg.is_multipart():
            layer_attachments = []
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
                        if not attachment_text:
                            attachment_text = extract_attachment_text(current_msg)
            attachments.append(layer_attachments)
        else:
            attachments.append([])

        layers.append(text.strip())
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

    return layers, metadata, attachments, attachment_text

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

# Function to classify email using GPT-2 with rule-based fallback for request type and sub-request type
def classify_email(text):
    logging.info("Classifying email content for request type and sub-request type...")
    if not text:
        logging.warning("Empty email text, returning default classification.")
        return "Unknown", "Unknown"

    prompt = f"""Classify the following email body into a request type and sub-request type from the list: {list(request_types.keys())}. Use this format:

    Request Type: [type]
    Sub-Request Type: [subtype]

    Email body:
    {text}
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=gpt2_max_length_input, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=gpt2_max_length_output,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"GPT-2 classification response: '{response}'")
    except Exception as e:
        logging.error(f"Error during GPT-2 classification: {e}")
        response = ""

    request_type_match = re.search(r"Request Type: (.*?)\n", response)
    sub_request_type_match = re.search(r"Sub-Request Type: (.*?)\n", response)
    request_type = request_type_match.group(1) if request_type_match else None
    sub_request_type = sub_request_type_match.group(1) if sub_request_type_match else None

    if not request_type or request_type not in request_types or not sub_request_type:
        logging.info("Falling back to rule-based classification...")
        text_lower = text.lower()
        for req_type, sub_types in request_types.items():
            if req_type.lower() in text_lower:
                request_type = req_type
                for sub_type in sub_types:
                    if sub_type.lower() in text_lower:
                        sub_request_type = sub_type
                        break
                if not sub_request_type:
                    sub_request_type = sub_types[0] if sub_types else "General"
                break
        else:
            request_type = "Unknown"
            sub_request_type = "Unknown"

    logging.info(f"Classification result - Request Type: {request_type}, Sub-Request Type: {sub_request_type}")
    return request_type, sub_request_type

# Function to detect primary intent
def detect_primary_intent(text, attachment_text=""):
    logging.info("Detecting primary intent for classification...")
    combined_text = text + "\n" + attachment_text
    
    prompt = f"""Determine the primary intent of the following email and its attachments. Choose from: {list(request_types.keys())} or "Unknown" if none match. Return only the intent:

    Email body and attachments:
    {combined_text}
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=gpt2_max_length_input, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=gpt2_max_length_output,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50
        )
        primary_intent = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logging.info(f"GPT-2 detected intent: '{primary_intent}'")
    except Exception as e:
        logging.error(f"Error during GPT-2 intent detection: {e}")
        primary_intent = ""

    if not primary_intent or primary_intent not in request_types:
        logging.info("Falling back to rule-based intent detection...")
        intent_scores = {}
        combined_text_lower = combined_text.lower()
        for request_type in request_types.keys():
            count = combined_text_lower.count(request_type.lower())
            if count > 0:
                intent_scores[request_type] = count
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "Unknown"

    logging.info(f"Final primary intent: {primary_intent}")
    return primary_intent

# Function to extract fields from attachment text
def extract_fields(attachment_text):
    logging.info("Extracting fields from attachments...")
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

    logging.info(f"Extracted fields: {fields}")
    return fields

# Function to detect duplicates
def detect_duplicates(email_chains, similarity_threshold=0.95):
    logging.info("Detecting duplicate email chains...")
    duplicates = []
    chain_details = []

    for type_id, email_id, layers, metadata, attachments, _ in email_chains:
        chain_signature = {
            "layers": layers,
            "metadata": metadata,
            "attachments": attachments
        }
        chain_details.append((type_id, email_id, chain_signature))

    def has_reply_indicator(metadata, layer_text, original_subject):
        current_subject = metadata["Subject"].strip().lower()
        original_subject = original_subject.strip().lower()
        subject_has_re = current_subject.startswith("re:") or " re: " in current_subject
        body_has_re = ("re: " + original_subject) in layer_text.lower()
        clean_current_subject = current_subject.replace("re:", "").strip()
        subject_matches = clean_current_subject == original_subject
        return (subject_has_re or body_has_re) and subject_matches

    for i, (type_id_i, email_id_i, chain_i) in enumerate(chain_details):
        for j, (type_id_j, email_id_j, chain_j) in enumerate(chain_details[i+1:], start=i+1):
            if len(chain_i["layers"]) != len(chain_j["layers"]):
                continue

            is_duplicate = True
            layer_similarities = []
            original_subject = chain_i["metadata"][0]["Subject"]

            for layer_idx in range(len(chain_i["layers"])):
                text_sim = compute_text_similarity(chain_i["layers"][layer_idx], chain_j["layers"][layer_idx])
                layer_similarities.append(text_sim)
                if text_sim < similarity_threshold:
                    is_duplicate = False
                    break

                meta_i = chain_i["metadata"][layer_idx]
                meta_j = chain_j["metadata"][layer_idx]
                
                is_reply_i = has_reply_indicator(meta_i, chain_i["layers"][layer_idx], original_subject)
                is_reply_j = has_reply_indicator(meta_j, chain_j["layers"][layer_idx], original_subject)
                
                if is_reply_i != is_reply_j:
                    is_duplicate = False
                    break
                    
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

# Function to store classification results
def store_classification(type_id, email_id, primary_intent, request_type, sub_request_type, fields):
    if primary_intent not in request_types:
        logging.warning(f"Invalid primary_intent '{primary_intent}' for type_{type_id}/email_{email_id}, using 'Unknown'")
        primary_intent = "Unknown"
    
    safe_intent = re.sub(r'[^a-zA-Z0-9\-]', '_', primary_intent)
    folder_path = f"{classifications_dir}/type_{type_id}/{safe_intent}/email_{email_id}"
    os.makedirs(folder_path, exist_ok=True)
    
    result = {
        "primary_intent": primary_intent,
        "request_type": request_type,
        "sub_request_type": sub_request_type,
        "fields": fields
    }
    
    try:
        with open(f"{folder_path}/classification.json", "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing classification.json for type_{type_id}/email_{email_id}: {e}")
        return
    
    eml_source_path = f"{email_files_dir}/type_{type_id}/email_{type_id}_{email_id}.eml"
    eml_dest_path = f"{folder_path}/email_{type_id}_{email_id}.eml"
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
        layers, metadata, attachments, attachment_text = extract_email_chain_details(msg)
        
        full_text = "\n".join(layers) if layers else ""
        email_chains.append((type_id, email_id, layers, metadata, attachments, attachment_text))

        # Classify based on primary intent
        primary_intent = detect_primary_intent(full_text, attachment_text)

        # Get latest request type and sub-request type from the last layer
        latest_text = layers[-1] if layers else ""
        request_type, sub_request_type = classify_email(latest_text)

        # Extract fields
        fields = extract_fields(attachment_text)

        # Store classification with all details
        store_classification(type_id, i, primary_intent, request_type, sub_request_type, fields)
        total_emails_processed += 1

# Detect duplicates
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