import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)  # Make sure the Flask app is initialized

@app.route('/')
def home():
    return "Face Similarity Detection App is Running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess an image (resize to 64x64, grayscale, and flatten)
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    return image.flatten(), image

# Function to calculate Manhattan distance
def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

# Function to calculate similarities
def calculate_similarities(face1, face2):
    euclidean_dist = np.linalg.norm(face1 - face2)
    cosine_sim = cosine_similarity([face1], [face2])[0][0]
    manhattan_dist = manhattan_distance(face1, face2)
    return euclidean_dist, cosine_sim, manhattan_dist

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "your_image" not in request.files or "father_image" not in request.files or "mother_image" not in request.files:
            return "Missing files!", 400

        # Save uploaded images
        your_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "your_image.jpg")
        father_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "father_image.jpg")
        mother_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "mother_image.jpg")

        request.files["your_image"].save(your_image_path)
        request.files["father_image"].save(father_image_path)
        request.files["mother_image"].save(mother_image_path)

        # Preprocess images
        your_face, your_image = preprocess_image(your_image_path)
        father_face, father_image = preprocess_image(father_image_path)
        mother_face, mother_image = preprocess_image(mother_image_path)

        # Calculate similarities
        euclidean_your_father, cosine_your_father, manhattan_your_father = calculate_similarities(your_face, father_face)
        euclidean_your_mother, cosine_your_mother, manhattan_your_mother = calculate_similarities(your_face, mother_face)

        cosine_father_percentage = round(cosine_your_father * 100, 2)
        cosine_mother_percentage = round(cosine_your_mother * 100, 2)
        cosine_father_percentage = round(cosine_your_father * 100, 2)
        cosine_mother_percentage = round(cosine_your_mother * 100, 2)


        # Determine closest match
        closer_euclidean = "Father" if euclidean_your_father < euclidean_your_mother else "Mother"
        closer_manhattan = "Father" if manhattan_your_father < manhattan_your_mother else "Mother"
        closer_cosine = "Father" if cosine_your_father > cosine_your_mother else "Mother"

        # Render results page
        return render_template(
            "results.html",
            your_image="uploads/your_image.jpg",
            father_image="uploads/father_image.jpg",
            mother_image="uploads/mother_image.jpg",
            euclidean_your_father=euclidean_your_father,
            cosine_your_father=cosine_father_percentage,
            manhattan_your_father=manhattan_your_father,
            euclidean_your_mother=euclidean_your_mother,
            cosine_your_mother=cosine_mother_percentage,
            manhattan_your_mother=manhattan_your_mother,
            closer_euclidean=closer_euclidean,
            closer_manhattan=closer_manhattan,
            closer_cosine=closer_cosine
        )


    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
