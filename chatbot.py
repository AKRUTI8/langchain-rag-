from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat():

    print("Groq Chatbot - Type 'quit' to exit\n")
    
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
            )
            
            assistant_message = response.choices[0].message.content
            
            messages.append({"role": "assistant", "content": assistant_message})
            
            print(f"Chatbot: {assistant_message}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
            messages.pop()

if __name__ == "__main__":
    chat()