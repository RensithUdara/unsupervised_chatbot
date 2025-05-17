import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            response TEXT NOT NULL,
            cluster INTEGER,
            feedback INTEGER,  -- 1 for thumbs-up, -1 for thumbs-down, NULL for no feedback
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_conversation(user_input, response, feedback=None):
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO conversations (user_input, response, feedback) VALUES (?, ?, ?)',
                   (user_input, response, feedback))
    conn.commit()
    conn.close()
def log_conversation(user_input, response):
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO conversations (user_input, response) VALUES (?, ?)', (user_input, response))
    conn.commit()
    conn.close()

def get_conversations():
    conn = sqlite3.connect('conversations.db')
    df = pd.read_sql_query('SELECT user_input, response, cluster FROM conversations', conn)
    conn.close()
    return df

def update_conversation_clusters(df):
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    for idx, row in df.iterrows():
        cursor.execute('UPDATE conversations SET cluster = ? WHERE user_input = ? AND response = ?',
                       (row['cluster'], row['user_input'], row['response']))
    conn.commit()
    conn.close()

def load_sample_data():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    sample_data = [
        ("Hello! How are you?", "Hi there! I'm doing great!"),
        ("What's up?", "Not much, just chilling. You?"),
        ("Tell me a joke.", "Why did the scarecrow become a motivational speaker? Because he was outstanding!"),
        ("How's the weather?", "It's sunny and bright today!"),
        ("What is AI?", "AI is tech that mimics human thinking—pretty cool, right?"),
        ("Can you help me with coding?", "Sure, I can give you some coding tips! What's the language?"),
        ("Good morning!", "Morning! Hope your day's off to a great start!"),
        ("Why is the sky blue?", "The sky looks blue because of Rayleigh scattering—light spreads in the atmosphere!"),
        ("What's your favorite color?", "I like all colors, but blue's pretty vibey!"),
        ("Tell me about space.", "Space is vast, with stars, planets, and mysteries like black holes!"),
        ("Bye for now!", "Catch you later!"),
        ("How do I learn Python?", "Start with basics: variables, loops, and functions. Try freeCodeCamp or Codecademy!"),
        ("What's 2 + 2?", "Easy, it's 4!"),
        ("Who made you?", "I was created by xAI, a team working on cool AI stuff!"),
        ("What's the time?", "I don't have a clock, but it's always time to chat!")
    ]
    cursor.executemany('INSERT OR IGNORE INTO conversations (user_input, response) VALUES (?, ?)', sample_data)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    load_sample_data()
    print("Database initialized with sample data.")