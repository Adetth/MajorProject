import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

def summarize_text(text, max_length=130, min_length=30, length_penalty=2.0):
    """
    Summarize the input text using a pre-trained BART model.

    Args:
        text (str): The input essay or text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
        length_penalty (float): Length penalty for controlling output length.

    Returns:
        str: The summarized text.
    """
    # Tokenize input and move to GPU (if available)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=2,  # Reduce beams for faster inference
        early_stopping=True
    )
    
    # Decode and return summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    essay = """
    Artificial intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks
    that require human intelligence. These include problem-solving, learning, reasoning, and natural language processing.
    AI is increasingly prevalent in modern society, powering applications such as recommendation systems, autonomous vehicles,
    and virtual assistants. The field has seen rapid advancements due to the availability of vast amounts of data and
    increased computational power. Despite its potential benefits, AI also raises ethical concerns, such as biases in decision-making,
    privacy issues, and the displacement of jobs.
    """
    print("Original Essay:\n")
    print(essay)
    print("\nSummary:\n")
    print(summarize_text(essay))
