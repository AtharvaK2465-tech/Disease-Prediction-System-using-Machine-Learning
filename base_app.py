from flask import Flask

# Initialize Flask app
app = Flask(__name__)

# Base route for testing
@app.route("/")
def home():
    return {"message": "Flask backend is running successfully!"}

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
