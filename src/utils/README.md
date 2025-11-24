"""
Machine Learning Models Project

This repository contains a collection of machine learning models implemented in Python.

Usage
-----

To train a model, navigate to the corresponding directory and run `python train.py`.

To evaluate a model, navigate to the corresponding directory and run `python evaluate.py`.

Contributing
------------

Contributions are welcome! Please fork this repository and submit a pull request.

License
-------

This project is licensed under the MIT License.

Authors
-------

* [Your Name](https://github.com/your-github-username)
"""

import os

def get_model_names():
    """Return a list of model names."""
    return [name for name in os.listdir() if os.path.isdir(name)]

def get_model_info(model_name):
    """Return information about a model."""
    return {
        'name': model_name,
        'description': 'A machine learning model',
        'author': 'Your Name',
        'license': 'MIT'
    }

if __name__ == '__main__':
    model_names = get_model_names()
    for model_name in model_names:
        print(get_model_info(model_name))