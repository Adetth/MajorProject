from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load a pre-trained BERT model fine-tuned on a sentence similarity task
model_name = "textattack/bert-base-uncased-MNLI"  # BERT fine-tuned on MNLI (Natural Language Inference)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def evaluate_essay(essay, answer_key):
    """
    Evaluate the essay by comparing it with the answer key using a BERT model.
    
    Args:
        essay (str): The student's essay.
        answer_key (str): The correct or ideal answer.
    
    Returns:
        float: A similarity score representing how well the essay matches the answer key.
    """
    # Prepare inputs
    inputs = tokenizer(essay, answer_key, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Use the "entailment" score as a similarity metric
    entailment_score = probabilities[0][2].item()  # Label 2 corresponds to "entailment" in MNLI
    return entailment_score

# Example usage
if __name__ == "__main__":
    student_essay = """
    Artificial intelligence is a technology that allows machines to perform tasks that typically require human intelligence.
    Examples include natural language processing, learning, and decision-making. AI is being applied in many areas such as
    healthcare, autonomous driving, and personal assistants. However, it raises ethical issues, including job displacement
    and privacy concerns.
    """
    answer_key = """
    Artificial intelligence involves the creation of systems capable of human-like tasks such as problem-solving, reasoning,
    and language understanding. Its applications include autonomous systems, virtual assistants, and data analysis. AI
    has ethical challenges, such as ensuring fairness, avoiding biases, and protecting privacy.
    """
    score = evaluate_essay(student_essay, answer_key)
    print(f"Essay Similarity Score: {score:.2f}")
    if score > 0.75:
        print("Grade: Excellent")
    elif score > 0.5:
        print("Grade: Good")
    elif score > 0.3:
        print("Grade: Needs Improvement")
    else:
        print("Grade: Poor")
