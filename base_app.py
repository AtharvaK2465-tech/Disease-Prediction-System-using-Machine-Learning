from flask import Flask

# Initialize Flask app
app = Flask(__name__)

# Base route to verify server
@app.route('/')
def home():
    return "Flask Backend is Running!"

if __name__ == "__main__":
    app.run(debug=True)
