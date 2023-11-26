from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from werkzeug.utils import secure_filename
from PIL import Image
import os
import torch
from torchvision import transforms
from training.model import Classifier
from training.utils import test_transform,CustomFolder

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'

state_dict = torch.load("training/w_models/best_model.pth", map_location=torch.device('cpu'))
model = Classifier(2)
model_dict = model.state_dict()
# Map the weights from the pre-trained model to your model
pretrained_dict = {k: v for k, v in zip(model_dict.keys(),state_dict.values())}
model_dict.update(pretrained_dict)
# Load the updated state dictionary into your model
model.load_state_dict(model_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ev=model.eval()

@app.route('/')
def index():
    return render_template('index.html')

    

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files:
            raise ValueError("No file part in the request")

        file = request.files['file']

        if file.filename == '':
            raise ValueError("No selected file")

        filename = secure_filename(file.filename)
        uploads_dir = 'uploads'
        os.makedirs(uploads_dir, exist_ok=True)  # Create 'uploads' directory if it doesn't exist
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)



        
        # Open the image and apply transformations
        predict_dataset = CustomFolder(root=file_path, transform=test_transform,has_labels=False)
        # Make prediction
        with torch.no_grad():
            probability,output,predicted_label= predict_dataset.predict_one(model)

        return render_template('result.html', filename=filename, predicted_label=predicted_label, probability=probability)

    except Exception as e:
        # Handle errors gracefully
        return render_template('error.html', error=str(e))
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5001)
