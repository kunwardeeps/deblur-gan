# What is this repo ?

This repository is a Keras implementation of [Deblur GAN](https://arxiv.org/pdf/1711.07064.pdf). It runs a Flask based webserver to get the output image from the trained model. Below is a sample result (from left to right: sharp image, blurred image, deblurred image)

![Sample results](./sample/results0.png)

# Installation

```
virtualenv venv -p python3
. venv/bin/activate
pip install -r requirements/requirements.txt
pip install -e .
```

# To start the Web Server

```
python deblur_image.py
```
