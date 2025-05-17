from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from database import get_conversations
import pandas as pd

def prepare_dataset():
    # Load conversations from database
    df = get_conversations()
    if df.empty:
        print("No data for fine-tuning.")
        return None

    # Format data for BlenderBot (input: user_input, output: response)
    data = {"input_text": df["user_input"].tolist(), "target_text": df["response"].tolist()}
    dataset = Dataset.from_dict(data)
    return dataset

def tokenize_data(dataset, tokenizer):
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

def fine_tune_model():
    # Load model and tokenizer
    model_name = "facebook/blenderbot-400M-distill"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare dataset
    dataset = prepare_dataset()
    if dataset is None:
        return

    # Tokenize data
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Model fine-tuned and saved!")

if __name__ == "__main__":
    fine_tune_model()