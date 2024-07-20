import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
from deeplab_model import DeepLabModel, drawSegment

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Update this path to the directory where your frozen_inference_graph.pb is located
model_path = "deeplabv3_mnv2_pascal_trainval"
MODEL = DeepLabModel(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'input_image' not in request.files or 'background_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        input_image = request.files['input_image']
        background_image = request.files['background_image']

        if input_image.filename == '' or background_image.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if input_image and allowed_file(input_image.filename) and background_image and allowed_file(background_image.filename):
            input_filename = secure_filename(input_image.filename)
            background_filename = secure_filename(background_image.filename)
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + input_filename)

            input_image.save(input_path)
            background_image.save(background_path)
            
            run_visualization(input_path, background_path, output_path)

            return redirect(url_for('uploaded_file', filename='output_' + input_filename))

    return render_template('index2.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def run_visualization(filepath, background_path, output_path):
    """Inferences DeepLab model and visualizes result."""
    try:
        print("Trying to open : " + filepath)
        jpeg_str = open(filepath, "rb").read()
        original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check file: ' + filepath)
        return

    print('running deeplab on image %s...' % filepath)
    resized_im, seg_map = MODEL.run(original_im)

    try:
        background = Image.open(background_path)
    except IOError:
        print('Cannot retrieve background image. Please check file: ' + background_path)
        return

    overlay = drawSegment(resized_im, seg_map)

    # Calculate position to center the overlay on the background
    bg_width, bg_height = background.size
    overlay_width, overlay_height = overlay.size
    position = ((bg_width - overlay_width) // 2, (bg_height - overlay_height) // 2)

    # Ensure the overlay is placed correctly in the center
    background.paste(overlay, position, overlay)
    background.save(output_path)
    print(f'Output image saved to {output_path}')

if __name__ == '__main__':
    app.run(debug=True)
