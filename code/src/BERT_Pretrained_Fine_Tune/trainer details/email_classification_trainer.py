# Import required libraries
try:
    from datasets import Dataset
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    import torch
    import pandas as pd
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure all required libraries are installed: datasets, transformers, torch, pandas")
    print("Run: pip install datasets transformers torch pandas")
    exit(1)

# Define the request types and sub-request types
REQUEST_TYPES = [
    "ADJUSTMENT",
    "AU TRANSFER",
    "CLOSING NOTICE",
    "COMMITMENT CHANGE",
    "FEE PAYMENT",
    "MONEY MOVEMENT - INBOUND",
    "MONEY MOVEMENT - OUTBOUND"
]

SUB_REQUEST_TYPES = {
    "ADJUSTMENT": [],
    "AU TRANSFER": [],
    "CLOSING NOTICE": [
        "REALLOCATION FEES",
        "AMENDMENT FEES",
        "REALLOCATION PRINCIPAL",
        "CASHLESS ROLL",
        "DECREASE"
    ],
    "COMMITMENT CHANGE": [
        "INCREASE",
        "ONGOING FEE"
    ],
    "FEE PAYMENT": [
        "LETTER OF CREDIT FEE",
        "PRINCIPAL",
        "INTEREST",
        "PRINCIPAL + INTEREST"
    ],
    "MONEY MOVEMENT - INBOUND": [
        "PRINCIPAL+INTEREST+FEE",
        "TIMEBOUND"
    ],
    "MONEY MOVEMENT - OUTBOUND": [
        "FOREIGN CURRENCY"
    ]
}

# Flatten sub-request types for classification
ALL_SUB_REQUEST_TYPES = []
for sub_list in SUB_REQUEST_TYPES.values():
    ALL_SUB_REQUEST_TYPES.extend(sub_list)

# Load the dataset with pandas to preprocess it
try:
    df = pd.read_csv("email_classification_dataset.csv")
except FileNotFoundError:
    print("Error: 'email_classification_dataset.csv' not found. Please generate the dataset first.")
    exit(1)

# Replace both NaN and None with the string "None"
df["sub_request_type"] = df["sub_request_type"].replace([pd.NA, None], "None")

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Verify dataset size
print(f"Dataset size: {len(dataset)}")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Map labels to indices
request_type_to_idx = {label: idx for idx, label in enumerate(REQUEST_TYPES)}
sub_request_type_to_idx = {label: idx for idx, label in enumerate(ALL_SUB_REQUEST_TYPES)}

# Preprocess the dataset
def preprocess_function(examples):
    # Tokenize the email content
    tokenized = tokenizer(examples["email_content"], truncation=True, padding="max_length", max_length=512)
    
    # Map request_type to label
    tokenized["request_type_label"] = [
        request_type_to_idx[rt] if rt is not None else -1 for rt in examples["request_type"]
    ]
    
    # Map sub_request_type to label (use -1 for "None")
    tokenized["sub_request_type_label"] = [
        sub_request_type_to_idx[srt] if srt != "None" else -1
        for srt in examples["sub_request_type"]
    ]
    
    return tokenized

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# --- Fine-tune the request_type_model ---

# Prepare the dataset for request_type_model by renaming the label column
request_dataset = tokenized_dataset.rename_column("request_type_label", "labels")

# Split into train and test sets (80% train, 20% eval)
train_size = int(0.8 * len(request_dataset))  # 128,000 samples for training
eval_size = len(request_dataset) - train_size  # 32,000 samples for evaluation
train_dataset = request_dataset.shuffle(seed=42).select(range(train_size))
eval_dataset = request_dataset.shuffle(seed=42).select(range(train_size, len(request_dataset)))

# Load the model for request type classification
request_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(REQUEST_TYPES)
)

# Define training arguments (use eval_strategy instead of evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./request_type_model",
    eval_strategy="epoch",  # Updated to eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define a compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    return {"accuracy": accuracy}

# Initialize the Trainer
trainer = Trainer(
    model=request_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
request_model.save_pretrained("./request_type_model")
tokenizer.save_pretrained("./request_type_model")

print("Fine-tuning completed for request_type_model. Model saved to './request_type_model'.")

# --- Fine-tune the sub_request_type_model ---

# Prepare the dataset for sub_request_type_model by renaming the label column
sub_request_dataset = tokenized_dataset.rename_column("sub_request_type_label", "labels")

# Filter dataset for sub-request type training (exclude "None" values)
sub_request_dataset = sub_request_dataset.filter(lambda x: x["labels"] != -1)

# Split into train and test sets for sub-request type (80% train, 20% eval)
train_size = int(0.8 * len(sub_request_dataset))
eval_size = len(sub_request_dataset) - train_size
sub_train_dataset = sub_request_dataset.shuffle(seed=42).select(range(train_size))
sub_eval_dataset = sub_request_dataset.shuffle(seed=42).select(range(train_size, len(sub_request_dataset)))

# Load the model for sub-request type classification
sub_request_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(ALL_SUB_REQUEST_TYPES)
)

# Define training arguments for sub-request model (use eval_strategy)
sub_training_args = TrainingArguments(
    output_dir="./sub_request_type_model",
    eval_strategy="epoch",  # Updated to eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer for sub-request model
sub_trainer = Trainer(
    model=sub_request_model,
    args=sub_training_args,
    train_dataset=sub_train_dataset,
    eval_dataset=sub_eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the sub-request model
sub_trainer.train()

# Save the sub-request model
sub_request_model.save_pretrained("./sub_request_type_model")
tokenizer.save_pretrained("./sub_request_type_model")

print("Fine-tuning completed for sub_request_type_model. Model saved to './sub_request_type_model'.")