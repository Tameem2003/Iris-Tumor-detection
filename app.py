from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
MODEL_PATH = "iris_tumor_cnn_model.h5"

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size
IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    """Preprocess the image for model prediction"""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image file was uploaded
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        try:
            # Save the uploaded file
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            logger.info(f"Saved uploaded file: {filepath}")

            # Preprocess the image
            img_array = preprocess_image(filepath)
            logger.info(f"Image shape after preprocessing: {img_array.shape}")

            # Make prediction
            prediction = model.predict(img_array)
            prediction_value = float(prediction[0][0])
            logger.info(f"Raw prediction value: {prediction_value}")

            # Adjust threshold based on model's behavior
            # You might need to adjust this threshold based on your model's performance
            threshold = 0.3  # Lowered threshold to detect tumors more sensitively
            result = "Tumorous" if prediction_value > threshold else "Healthy"
            
            confidence = prediction_value if result == "Tumorous" else (1 - prediction_value)
            confidence_percentage = f"{confidence * 100:.2f}%"
            
            logger.info(f"Final prediction: {result} with confidence {confidence_percentage}")

            # Get base64 encoded image
            import base64
            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Remove the uploaded file
            os.remove(filepath)
            
            return render_template("result.html", 
                                 result=result, 
                                 confidence=confidence_percentage,
                                 raw_value=f"{prediction_value:.4f}",
                                 image=img_data)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Error processing image: {str(e)}", 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
