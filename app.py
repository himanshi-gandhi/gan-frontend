import numpy as np
import tensorflow as tf
from flask import Flask, render_template, send_file, make_response
from PIL import Image
import io

# Load the trained generator model
generator = tf.keras.models.load_model("pickle/FashionGenerator.h5", compile=False)

app = Flask(__name__)

def generate_fashion_image():
    noise_dim = 128
    noise = np.random.normal(0, 1, (1, noise_dim))
    generated_image = generator.predict(noise)[0, :, :, 0]
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)
    return generated_image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_image")
def generate_image():
    img_array = generate_fashion_image()
    img = Image.fromarray(img_array, mode='L')

    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    response = make_response(img_io.read())
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
    return response

if __name__ == "__main__":
    app.run(debug=True)
