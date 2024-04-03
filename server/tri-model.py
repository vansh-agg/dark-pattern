from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertForSequenceClassification,BertTokenizer,XLNetTokenizer,AutoTokenizer,XLNetForSequenceClassification
from torch.nn.functional import softmax
import torch
import re

model_path = r"C:\Users\Vansh Aggarwal\OneDrive\Pictures\Documents\ML\deep learning\fraud-or-not\public\models\roberta_fine_tune"

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)


max_seq_length = 512

def preprocess_text(text):
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, truncation=True)))
    return tokens

def predict_dark_patterns(input_text):
    input_ids = tokenizer.encode(preprocess_text(input_text), return_tensors='pt', max_length=max_seq_length, truncation=True)

    with torch.no_grad():
        outputs = model(input_ids)

    probs = softmax(outputs.logits, dim=1).squeeze()
    predicted_category = torch.argmax(probs).item()

    return predicted_category, probs[predicted_category].item()

def count_dark_patterns(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Remove whitespaces
    text_content = re.sub(r'\s+', ' ', text_content)

    # Mapping category names to numeric labels
    category_mapping = {"Urgency": 0, "Not Dark Pattern": 1, "Scarcity": 2, "Misdirection": 3, "Social Proof": 4,
                        "Obstruction": 5, "Sneaking": 6, "Forced Action": 7}

    dark_patterns = {category: 0 for category in category_mapping}
    total_sentences = 0

    sentences = re.split(r'[.!?]', text_content)
    darkdata=[]
    for sentence in sentences:
        if not sentence.strip():
            continue

        category, _ = predict_dark_patterns(sentence)
        category_name = next(key for key, value in category_mapping.items() if value == category)

        # Exclude "Not Dark Pattern" category
        if category_name != "Not Dark Pattern":
            dark_patterns[category_name] += 1
            darkdata.append(sentence)
        total_sentences += 1

    return dark_patterns, total_sentences,darkdata

result, total_sentences,darksentences = count_dark_patterns('output.txt')

for category, count in result.items():
    if category != "Not Dark Pattern":
        print(f"{category}: {count} occurrences")

percentage = sum(result.values()) / total_sentences * 100
print(f"Percentage of Total Dark Patterns: {percentage:.2f}%")

