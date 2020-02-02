import numpy as np
from PIL import Image
import click
import os
import glob


from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image

from keras.backend import clear_session

from flask import Flask, flash, request, redirect, url_for, send_file, render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def deblur(weight_path, input_dir, output_dir):
	g = generator_model()
	g.load_weights(weight_path)
	for image_name in os.listdir(input_dir):
	    image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
	    x_test = image
	    generated_images = g.predict(x=x_test)
	    generated = np.array([deprocess_image(img) for img in generated_images])
	    x_test = deprocess_image(x_test)
	    for i in range(generated_images.shape[0]):
	        img = generated[i, :, :, :]
	        im = Image.fromarray(img.astype(np.uint8))
	        im.save(os.path.join(output_dir, image_name))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@click.command()
@click.option('--weight_path', help='Model weight')
@click.option('--input_dir', help='Image to deblur')
@click.option('--output_dir', help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):
    return deblur(weight_path, input_dir, output_dir)

@app.route("/")
@cross_origin()
def upload_form():
	return render_template('home.html')

@app.route("/clear")
@cross_origin()
def clear_files():
	clear_session()
	files = glob.glob('images/*')
	for f in files:
		os.remove(f)
	files = glob.glob('output/*')
	for f in files:
		os.remove(f)
	return 'success'

@app.route('/', methods=['POST'])
@cross_origin()
def upload_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		flash('No selected file')
		return {'error': 'No selected file'}, 404
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		deblur('generator.h5', 'images', 'output')
		clear_session()
		return send_file('output/' + filename)
	return {'error': 'Generic Error!'}, 404

if __name__ == "__main__":
	# app.run()
    app.run(host= '0.0.0.0')