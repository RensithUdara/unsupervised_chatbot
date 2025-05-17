from transformers import pipeline

# Initialize the conversational model
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")

def get_response(user_input):
    # Generate a response
    response = chatbot(user_input)
    return response[-1]['generated_text']

# Test the chatbot
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print(f"Bot: {response}")