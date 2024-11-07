from flask import Flask

app = Flask(__name__)

@app.route('/')  # This defines the route for the home page
def home():
    return "Welcome to the Potato Leaf Disease Classification API!"

# Existing code, like model loading and other routes, goes here.

if __name__ == "__main__":
    app.run(debug=True)
