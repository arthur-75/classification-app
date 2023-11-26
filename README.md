# Image Seeker

You write a description, Image Seeker gives you the image that fits the best with it. The image will come from a 1000-images-large dataset (cf. *2. Data*).

###### Using Dcoker 
* Open Docker, open the app if you are using Mac/wido
* run the following command to build the Docker image in your terminale  `docker build -t image-classifier-app .` note you should be inside trimble folder 
* After building the Docker image, you can run a container from it using the following command: `docker run -p 8080:5001 image-classifier-app`
* Open your web browser and go to `http://localhost:5000`
###### 1. Anaconda environment

To set up the environment, execute in a terminal the followings:

* `conda create -n trimble`
* `conda activate trimble`
* `conda install trimble`
* `pip install -r requirements-dev.txt`

###### 2. Run app

* `cd image_seeker` 
* `python -m streamlit run engine.py`

###### 3. Data

``imagenet-sample``, a small sample of images from the ImageNet 2012 dataset. The dataset contains 1,000 images, one randomly chosen from each class of the validation split of the ImageNet 2012 dataset.

###### 4. Model

The model used in this package is [CLIP](https://openai.com/research/clip), proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP (Contrastive Language-Image Pre-Training) is a zero-shot neural network trained on a variety of (image, text) pairs.
The version implemented here is `clip-vit-base-patch32`. The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.



