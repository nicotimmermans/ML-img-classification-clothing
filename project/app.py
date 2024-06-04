from flask import Flask, render_template, request, jsonify
import numpy as np
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the saved model
model = load_model("saved_model.h5")
model.summary()

# Preprocess the input image for prediction
def preprocess_image(img_path):
    try:
        # Download the image from the URL
        response = requests.get(img_path)
        response.raise_for_status()  # Check if the request was successful
        img_data = BytesIO(response.content)

        # Load the image
        img = Image.open(img_data)
        # img = img.convert("RGB")
        img = img.resize((28, 28))  # Adjust the size as needed

        # Convert PIL image to numpy array
        img_array = np.array(img)

        # Normalize the image
        # img_array = img_array / 255.0

        # Expand dimensions to match the model's expected input shape
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
    
# Flask route to serve HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Flask route to make predictions based on image URLs
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        if image_url is None:
            return jsonify({"error": "Invalid request. Missing 'image_url' parameter."}), 400

        # Preprocess the image for prediction
        img_array = preprocess_image(image_url)

        # Make prediction using the loaded model
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class = np.argmax(predictions)

        # Map the class index to class name
        class_name = class_names[predicted_class]

        # Return the predicted class name
        return jsonify({"class_name": class_name}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
            
if __name__ == "__main__":
    app.run(debug=True)