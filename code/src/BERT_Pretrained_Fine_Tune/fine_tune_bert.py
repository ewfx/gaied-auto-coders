import json
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load configuration
with open("config_advanced.json", "r") as config_file:
    config = json.load(config_file)

request_types_config = config["request_types"]
REQUEST_TYPES = list(request_types_config.keys())
SUB_REQUEST_TYPES = {key: value for key, value in request_types_config.items()}
ALL_SUB_REQUEST_TYPES = []
for sub_list in SUB_REQUEST_TYPES.values():
    ALL_SUB_REQUEST_TYPES.extend(sub_list)
ALL_SUB_REQUEST_TYPES = list(dict.fromkeys(ALL_SUB_REQUEST_TYPES))

# Create a synthetic dataset (replace with your actual dataset)
# Example: List of (text, request_type, sub_request_type) tuples
data = []
for i in range(1000):  # Generate 1000 synthetic examples
    request_type = random.choice(REQUEST_TYPES)
    sub_request_type = random.choice(SUB_REQUEST_TYPES[request_type])
    text = f"Request Type: {request_type}\nSub-Request Type: {sub_request_type}\nPlease process this request.\nDeal ID: {random.randint(1000, 9999)}\nAmount: ${random.randint(10000, 50000)}\nExpiration Date: 2025-12-31"
    data.append({
        "text": text,
        "request_label": REQUEST_TYPES.index(request_type),
        "sub_request_label": ALL_SUB_REQUEST_TYPES.index(sub_request_type)
    })

# Convert to Dataset
dataset = Dataset.from_list(data)
train_data, eval_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "request_label", "sub_request_label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "request_label", "sub_request_label"])

# Fine-tune request_type model
request_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(REQUEST_TYPES))
training_args = TrainingArguments(
    output_dir="./request_type_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=request_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda eval_pred: {"accuracy": (eval_pred.predictions.argmax(1) == eval_pred.label_ids).mean()}
)

trainer.train()
request_model.save_pretrained("./request_type_model")
tokenizer.save_pretrained("./request_type_model")

# Fine-tune sub_request_type model
sub_request_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(ALL_SUB_REQUEST_TYPES))
training_args.output_dir = "./sub_request_type_model"

trainer = Trainer(
    model=sub_request_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda eval_pred: {"accuracy": (eval_pred.predictions.argmax(1) == eval_pred.label_ids).mean()}
)

trainer.train()
sub_request_model.save_pretrained("./sub_request_type_model")
tokenizer.save_pretrained("./sub_request_type_model")

print("Fine-tuning complete. Models saved to ./request_type_model and ./sub_request_type_model.")