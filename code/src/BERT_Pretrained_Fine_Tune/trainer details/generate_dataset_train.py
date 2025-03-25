import pandas as pd
import random

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

# Templates for generating synthetic email content
TEMPLATES = {
    "ADJUSTMENT": [
        "Please process an adjustment for the account {account_id}.",
        "We request an adjustment due to recent changes in {account_id}.",
        "Adjustment needed for {account_id} as per the latest update."
    ],
    "AU TRANSFER": [
        "Initiate an AU transfer for {account_id} to the new account.",
        "We need an AU transfer for {account_id} effective immediately.",
        "Please handle the AU transfer for {account_id} this week."
    ],
    "CLOSING NOTICE": {
        "REALLOCATION FEES": [
            "This is a closing notice regarding reallocation fees for {account_id}.",
            "Closing notice: reallocation fees are due for {account_id}.",
            "Please note the closing notice for reallocation fees on {account_id}."
        ],
        "AMENDMENT FEES": [
            "Closing notice for amendment fees related to {account_id}.",
            "This email serves as a closing notice for amendment fees on {account_id}.",
            "Closing notice: amendment fees are applicable for {account_id}."
        ],
        "REALLOCATION PRINCIPAL": [
            "Closing notice for reallocation principal on {account_id}.",
            "This is a closing notice regarding the reallocation principal for {account_id}.",
            "Please process the closing notice for reallocation principal of {account_id}."
        ],
        "CASHLESS ROLL": [
            "Closing notice for a cashless roll on {account_id}.",
            "This email is a closing notice for the cashless roll of {account_id}.",
            "Closing notice: cashless roll is scheduled for {account_id}."
        ],
        "DECREASE": [
            "Closing notice for a decrease in {account_id}.",
            "This is a closing notice regarding the decrease for {account_id}.",
            "Please note the closing notice for the decrease on {account_id}."
        ]
    },
    "COMMITMENT CHANGE": {
        "INCREASE": [
            "Requesting a commitment change to increase the limit for {account_id}.",
            "Commitment change: please increase the commitment for {account_id}.",
            "We need a commitment change to increase the amount for {account_id}."
        ],
        "ONGOING FEE": [
            "Commitment change for an ongoing fee on {account_id}.",
            "Please process a commitment change for the ongoing fee of {account_id}.",
            "Commitment change: ongoing fee adjustment for {account_id}."
        ]
    },
    "FEE PAYMENT": {
        "LETTER OF CREDIT FEE": [
            "Fee payment required for the letter of credit fee on {account_id}.",
            "Please process the fee payment for the letter of credit fee of {account_id}.",
            "Fee payment for letter of credit fee is due for {account_id}."
        ],
        "PRINCIPAL": [
            "Fee payment for the principal amount on {account_id}.",
            "Please handle the fee payment for the principal of {account_id}.",
            "Fee payment: principal is due for {account_id}."
        ],
        "INTEREST": [
            "Fee payment for the interest on {account_id}.",
            "Please process the fee payment for interest on {account_id}.",
            "Fee payment for interest is required for {account_id}."
        ],
        "PRINCIPAL + INTEREST": [
            "Fee payment for principal + interest on {account_id}.",
            "Please process the fee payment for principal + interest of {account_id}.",
            "Fee payment: principal + interest is due for {account_id}."
        ]
    },
    "MONEY MOVEMENT - INBOUND": {
        "PRINCIPAL+INTEREST+FEE": [
            "Money movement - inbound for principal+interest+fee on {account_id}.",
            "Please process the money movement - inbound for principal+interest+fee of {account_id}.",
            "Money movement - inbound: principal+interest+fee for {account_id}."
        ],
        "TIMEBOUND": [
            "Money movement - inbound with timebound processing for {account_id}.",
            "Please handle the money movement - inbound with timebound for {account_id}.",
            "Money movement - inbound: timebound transaction for {account_id}."
        ]
    },
    "MONEY MOVEMENT - OUTBOUND": {
        "FOREIGN CURRENCY": [
            "Money movement - outbound in foreign currency for {account_id}.",
            "Please process the money movement - outbound in foreign currency for {account_id}.",
            "Money movement - outbound: foreign currency transaction for {account_id}."
        ]
    }
}

# Generate the dataset
data = []
num_samples_per_category = 40  # Number of samples per request type or sub-request type

for request_type in REQUEST_TYPES:
    if not SUB_REQUEST_TYPES[request_type]:  # No sub-request types
        for _ in range(num_samples_per_category):
            account_id = f"ACC{random.randint(1000, 9999)}"
            email_content = random.choice(TEMPLATES[request_type]).format(account_id=account_id)
            data.append({
                "email_content": email_content,
                "request_type": request_type,
                "sub_request_type": -1
            })
    else:  # Has sub-request types
        for sub_request_type in SUB_REQUEST_TYPES[request_type]:
            for _ in range(num_samples_per_category):
                account_id = f"ACC{random.randint(1000, 9999)}"
                email_content = random.choice(TEMPLATES[request_type][sub_request_type]).format(account_id=account_id)
                data.append({
                    "email_content": email_content,
                    "request_type": request_type,
                    "sub_request_type": sub_request_type
                })

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("email_classification_dataset.csv", index=False)

print("Dataset generated and saved to 'email_classification_dataset.csv'")
print(f"Total samples: {len(df)}")
print("\nSample of the dataset:")
print(df.head())