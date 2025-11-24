import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv()

# Check if API key exists to prevent errors later
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)

# --- 2. FLASK APP INITIALIZATION ---
app = Flask(__name__)
# This allows your frontend (even on a different address) to talk to this backend
CORS(app) 

# --- 3. GEMINI MODEL SETUP ---
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Note: Ensure "gemini-2.0-flash" is available in your region/account.
# If it fails, try changing model_name to "gemini-1.5-flash"
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="You are Nexus AI. Analyze the user's question. If simple, answer in 1 sentence. If complex, explain using ONE simple analogy and keep it under 3 sentences. Do not use filler phrases.",
)

# --- 4. CONVERSATION HISTORY MANAGEMENT ---
# We use a dictionary to store chat histories for different users (sessions)
chat_sessions = {}

# --- 5. API ENDPOINTS ---

@app.route('/')
def home():
    """
    Serve the index.html file.
    """
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests from the frontend.
    """
    try:
        # Get the user's message and a unique session ID from the request
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_input = data['message']
        session_id = data.get('session_id', 'default_session') # Use a default if not provided

        # Get or create a chat session for the user
        if session_id not in chat_sessions:
            chat_sessions[session_id] = model.start_chat(history=[])
        
        chat_session = chat_sessions[session_id]

        # Send the message to the Gemini API
        response = chat_session.send_message(user_input)
        
        # Return the model's response to the frontend
        return jsonify({'reply': response.text})

    except Exception as e:
        print(f"Error: {e}")
        # Return a generic error message if something goes wrong
        return jsonify({'error': str(e)}), 500

# --- 6. RUN THE FLASK APP ---
if __name__ == '__main__':
    # This will start a local development server
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)