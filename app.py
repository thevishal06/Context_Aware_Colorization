
from flask import Flask, request, render_template, redirect, url_for
from model import load_model, predict_colorization
import numpy as np
import cv2

app = Flask(__name__)
model = load_model()  # Load the model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Read and process the image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    colorized_image = predict_colorization(model, image)

    # Convert the output back to an image format for display
    output_image = (colorized_image[0] * 255).astype(np.uint8)  # Rescale to 0-255

    # Save or render the output image
    cv2.imwrite('static/output.jpg', output_image)

    return render_template('result.html', output_image='static/output.jpg')

if __name__ == '__main__':
    app.run(debug=True)
