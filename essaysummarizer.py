from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model (BERT for text summarization is better achieved with BART or similar models)
model_name = "facebook/bart-base"  # A BART model fine-tuned for summarization
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

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
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
if __name__ == "__main__":
    essay = """
    Artificial intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks
    that require human intelligence. These include problem-solving, learning, reasoning, and natural language processing.
    AI is increasingly prevalent in modern society, powering applications such as recommendation systems, autonomous vehicles,
    and virtual assistants. The field has seen rapid advancements due to the availability of vast amounts of data and
    increased computational power. Despite its potential benefits, AI also raises ethical concerns, such as biases in decision-making,
    privacy issues, and the displacement of jobs.
    """
    summary = summarize_text(essay)
    print("Original Essay:")
    print(essay)
    print("\nSummary:")
    print(summary)
