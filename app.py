from flask import Flask, request, jsonify, render_template
from chatbot import get_response
from database import init_db
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response = get_response(user_input)
    return jsonify({"response": response, "message_id": get_latest_message_id()})

@app.route('/feedback', methods=['POST'])
def feedback():
    message_id = request.json.get('message_id')
    feedback = request.json.get('feedback')  # 1 for thumbs-up, -1 for thumbs-down
    update_feedback(message_id, feedback)
    return jsonify({"status": "Feedback recorded"})

def get_latest_message_id():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(id) FROM conversations')
    message_id = cursor.fetchone()[0]
    conn.close()
    return message_id

def update_feedback(message_id, feedback):
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE conversations SET feedback = ? WHERE id = ?', (feedback, message_id))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)