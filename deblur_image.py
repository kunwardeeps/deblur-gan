import numpy as np
from PIL import Image
import click
import os


from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image

from flask import Flask, flash, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
def upload_form():
	return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/', methods=['POST'])
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
		return send_file('output/' + filename)
	return {'error': 'Generic Error!'}, 404

if __name__ == "__main__":
    app.run(host= '0.0.0.0')