import pandas as pd
import numpy as np
import random
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

# Load the dataset
LENGTH = 256
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

df = pd.DataFrame(dataset)
df['input'] = df['input'].astype(str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

# Define constants
max_length = tokenizer.model_max_length

def perturb_sentence(sentence, mask_prob=0.15):
    try:
        # Tokenize and check the length of the sentence
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]  # Truncate if too long
        masked_tokens = [tokenizer.mask_token if random.random() < mask_prob else token for token in tokens]

        masked_sentence = tokenizer.convert_tokens_to_string(masked_tokens)
        inputs = tokenizer(masked_sentence, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = outputs.logits
        predicted_tokens = []

        for idx, token in enumerate(masked_tokens):
            if token == tokenizer.mask_token:
                if idx < predictions.shape[1]:  # Ensure idx is within bounds
                    predicted_token_id = torch.argmax(predictions[0, idx]).item()
                    predicted_token = tokenizer.decode(predicted_token_id)
                    predicted_tokens.append(predicted_token)
                else:
                    predicted_tokens.append(token)  # Fallback if out of bounds
            else:
                predicted_tokens.append(token)

        return tokenizer.convert_tokens_to_string(predicted_tokens)

    except Exception as e:
        print(f"Error perturbing sentence: {sentence}, Error: {e}")
        return sentence  # Return the original sentence if error occurs

# Apply perturbation to the dataset
original_sentences = []
perturbed_sentences = []

for sentence in df['input']:
    if len(tokenizer.tokenize(sentence)) <= max_length:
        original_sentences.append(sentence)
        perturbed_sentence = perturb_sentence(sentence)
        perturbed_sentences.append(perturbed_sentence)
    else:
        print(f"Sentence too long and skipped: {sentence}")

# Save the results to CSV files
df_original = pd.DataFrame(original_sentences, columns=['input'])
df_original.to_csv("WikiMIA_prompt.csv", index=False)

df_perturbed = pd.DataFrame(perturbed_sentences, columns=['input'])
df_perturbed.to_csv("WikiMIA_perturbed.csv", index=False)

# Add labels to the CSV files
def add_labels(df, num_labels_per_class):
    n = len(df)
    labels = np.concatenate([np.zeros(num_labels_per_class), np.ones(n - num_labels_per_class)])
    np.random.shuffle(labels)
    df['label'] = labels
    return df

# Add labels
num_labels_per_class = len(original_sentences) // 2
df_original = add_labels(df_original, num_labels_per_class)
df_perturbed = add_labels(df_perturbed, num_labels_per_class)

# Save the labeled data
df_original.to_csv("WikiMIA_prompt_labeled.csv", index=False)
df_perturbed.to_csv("WikiMIA_perturbed_labeled.csv", index=False)

# Print DataFrame heads for verification
print(df_original.head(20))
print(df_perturbed.head(20))
