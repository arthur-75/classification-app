# Classification-app

## Run

### Using Dcoker 
(It would take 10 min)
* Open Docker, open the app if you are using Mac/Windows
* run the following command to build the Docker image in your terminale  `docker build -t image-classifier-app .` note you should be inside trimble folder 
* After building the Docker image, you can run a container from it using the following command: `docker run -p 8080:5001 image-classifier-app`
* Open your web browser and go to `http://127.0.0.1:5001`

### Anaconda environment

To set up the environment, execute in a terminal the followings:

* `conda create -n trimble`
* `conda activate trimble`
* `conda install trimble`
* `pip install -r requirements-dev.txt`

### Run app
* `python app.py` to run the application web
* then you go to `http://127.0.0.1:5001`

### Run script Main 
If you would like to run a folder of images you can do the command below and then you would get file .csv with "file","pred","label" : 
* `python main.py` to run python the script (optional : add two arguments `python main.py --input_folder training/dataset/test_images --output_csv predictions.csv` )

### Or use PYIJB to make predictions and / or training a new model
* training_prediction.ipynb



## Files 
```bash
.
├── main.py "(main file with two argParse 'input_folder' and 'output_csv'  )"
├── Dockerfile "(to run the application without conda)"
├── app.py  "(to run the app web)"
├── Challenge classification.pdf report with explication 
├── images_Analysis.ipynb  
├── requirements-dev.txt  "(requirements for Conda environment)"
├── requirements.txt "(requirements for Python conda)"
├── static  "(styles for the webAPP)"
│   └── styles.css
├── templates  "(HTML for the webAPP)"
│   ├── error.html
│   ├── index.html
│   └── result.html
├── training  "(in this file we can find the data set, model, weights, and utils functions)"
│   ├── dataset
│   │   ├── test_images
│   │   │   ├── ...
│   │   └── train_images
│   │       ├── fields
│   │       │   ├── ...
│   │       └── roads
│   │           ├── ...
│   ├── model.py
│   ├── utils.py
│   └── w_models  "(model's weights)"
│       ├── best_model.pth "(the one that we are using)"
│       ├── model-7.pth
│       └── w_model.pth
├── training_prediction.ipynb "(you make predictiions and training here )"
└── uploads "(folder saved automaticlly the image that apears on the web APP )"
```

## Data

``training/dataset/train_images``, a small sample of images. The dataset contains 153 images,  randomly chosen from each class of the train, validation, test split.
* Train: To train and prepare the model. 
* Valid: To validate which model and  strategies to choose.  
* Test: For Stress test/final validation.(I didn't use your folder test_images)
Train 60% ( “91 observations”), Valid 19% (“30 obs”) and  Test 21% (”32 obs”)

The images undergo transformation and augmentation using the following steps:
* Resize: 224x224
* Random Horizontal Flip
* Random Rotation
* Colour Jitter: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2 Normalisation: Mean = 456, Std = 224
* Additionally, a personalised segmentation process is applied, focusing on extracting colours such as green yellow, grey, and red while masking other colours. For an example of this segmentation function, refer to the photo below and check the "add_segmentation(self, image)" function in the utils module.
* While other techniques, such as edge detection and LBP Texture Analysis, were explored, time constraints prevented their integration into the production pipeline.


## Model

CNN Hybrid Approach:
Rather than building a CNN model from scratch, we leverage Transfer Learning. Specifically, we harness the GoogleNet architecture along with its pre-trained weights. This approach not only conserves resources but also avoids duplicating the efforts of other accomplished scientists. (Note: Two additional architectures have also been tested.)


### Input
Two inputs are considered for the model:
Segmented and Preprocessed Images: Injected directly into Googlenet.
Additional Information: Mean and variance information is injected into a dense layer. 


###  Result 
The CNN hybrid model has demonstrated superior performance compared to both the baseline and SVM, particularly regarding the recall and precision metrics for fields. But a potential limitation of the model concerning the detection of roads in scenarios where the road is unclear, and there is a significant presence of green/yellow pixels, the model may encounter challenges in accurately identifying roads. This situation highlights a specific condition under which the model may falter, emphasising the need for further refinement or consideration of alternative strategies, especially in environments with pronounced colour variations.
