# Requirement

- Python 3.8 - 3.11
- Pip

# Datasets:

- 43 folders represent 43 classes.
- Each folder contains multiple images (png format) of a traffic sign
- Each folder contains a csv file which has information of each images in the same directory

# Set up:

- Run `pip install -r setup.txt` on your shell
- Download [dataset](https://bit.ly/traffic-signs-recognizer-dataset) 
- Extract the dataset into `./dataset` 
- Run the notebook

# Demo:

- Run `python cli.py model_path your_image_path` to receive the prediction of our pre-trained model in your terminal.
- Run `python gui.py model_path` to open GUI app.

# Re-train the model:

- Download the dataset
- Take a look at `cnn.ipynb` and re-run

# Pre-trained model:
- If you do not want to re-train, you can download our [model](https://bit.ly/traffic-signs-recognition-models)