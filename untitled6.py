# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 08:52:14 2024

@author: Vik
"""

from flask import Flask, request, render_template_string, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Încărcam modelul salvat anterior
model = load_model('cifar10_model.h5')

# clasele CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Șablon HTML simplu pentru încărcarea fișierelor
HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Upload an Image</title>
</head>
<body>
  <h2>Incarca o imagine pentru clasificare cu modelul CIFAR-10</h2>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  {{message}}
</body>
</html>
'''

def process_image(file_stream):
    """Procesam un fișier de tip Image încărcat pentru predicție."""
    image = Image.open(file_stream).convert('RGB')
    # Decupam si redimensionam in forma de pătrat
    width, height = image.size
    new_size = min(width, height)
    image = image.crop(((width - new_size) / 2, (height - new_size) / 2, (width + new_size) / 2, (height + new_size) / 2))
    image = image.resize((32, 32))
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # O dimensiune noua de genul (1, 32, 32, 3)

#Butonul de upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            image_array = process_image(file.stream)
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            message = f"Clasa obiectului: {class_names[predicted_class]} cu confidenta {confidence:.2f}%"
            return render_template_string(HTML_TEMPLATE, message=message)
        return render_template_string(HTML_TEMPLATE, message="Incarcati o poza.")
    return render_template_string(HTML_TEMPLATE, message="")

if __name__ == '__main__':
    app.run(debug=True)
