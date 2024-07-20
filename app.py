import os
import requests
import json
import base64
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
from io import BytesIO 
import io
import google.generativeai as genai
from deeplab_model import DeepLabModel, drawSegment

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# DeepLab model setup
model_path = "deeplabv3_mnv2_pascal_trainval"
MODEL = DeepLabModel(model_path)

# Google Generative AI setup
API_URL1 = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
API_URL2 = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
headers = {"Authorization": "Bearer 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"}

api_key = "AIzaSyDnhDLrV74ffbfHx7fEto9Mf_SEquohqao"
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_model = genai.GenerativeModel('gemini-pro')
chat = chat_model.start_chat(history=[])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/chat')
def chat_page():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    message = request.form.get('message')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    response = chat.send_message(message)
    
    return jsonify({'response': response.text})

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    return jsonify({'history': chat.history})

@app.route('/analyze')
def index():
    return render_template('index1.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    prompt = request.form.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is missing"}), 400
    
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
    image.save(image_path)

    uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")
    
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [uploaded_file, prompt],
            }
        ]
    )

    response = chat_session.send_message(prompt)

    if response:
        result = response.text
    else:
        result = "Error occurred while analyzing the image"

    return jsonify({"result": result})

def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

@app.route("/image", methods=["GET", "POST"])
def generate_image():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400

        payload = {"inputs": prompt}
        response = requests.post(API_URL1, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                image.save("static/output_image.png")
                return jsonify({"image_url": "/static/output_image.png"})
            except IOError as e:
                return jsonify({"error": f"Unable to process the image. Details: {e}"}), 500
        else:
            return jsonify({"error": f"API request failed with status code {response.status_code}"}), 500
    else:
        return render_template("image.html")

def query(payload):
    response = requests.post(API_URL2, headers=headers, json=payload)
    return response.json()

@app.route('/upload', methods=['GET', 'POST'])
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

    bg_width, bg_height = background.size
    overlay_width, overlay_height = overlay.size
    position = ((bg_width - overlay_width) // 2, (bg_height - overlay_height) // 2)

    background.paste(overlay, position, overlay)
    background.save(output_path)
    print(f'Output image saved to {output_path}')

if __name__ == '__main__':
    app.run(debug=True)
