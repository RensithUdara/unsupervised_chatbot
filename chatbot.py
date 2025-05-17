from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

# Initialize the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# File to store conversations
CONVERSATION_LOG = "conversations.csv"

def log_conversation(user_input, response):
    # Create or append to CSV
    data = {"user_input": user_input, "response": response}
    df = pd.DataFrame([data])
    if os.path.exists(CONVERSATION_LOG):
        df.to_csv(CONVERSATION_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(CONVERSATION_LOG, index=False)

def get_response(user_input):
    # Encode the input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log the conversation
    log_conversation(user_input, response)
    
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print(f"Bot: {response}")