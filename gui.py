import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import keras
import cv2
import numpy as np
import sys

keras.utils.set_random_seed(42)

model = keras.models.load_model(sys.argv[1])


def make_prediction(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))

    test = np.array([img])
    res = model.predict(test, verbose=2)
    pred = np.argmax(res, axis=1)
    return pred[0]


class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic signs prediction")

        self.image_path = None

        self.input_label = tk.Label(root, text="Input")
        self.input_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.output_label = tk.Label(root, text="Output")
        self.output_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.upload_button = tk.Button(
            root, text="Upload Image", command=self.upload_image
        )
        self.predict_button = tk.Button(
            root, text="Predict", command=self.display_prediction
        )
        self.upload_button.grid(row=1, column=0, pady=10, sticky="ew")
        self.predict_button.grid(row=1, column=1, pady=10, sticky="ew")

        # Configure rows and columns to expand with the window
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")],
        )

        if file_path:
            self.image_path = file_path
            self.display_image(self.image_path, self.input_label)

    def display_image(self, path, label):
        image = Image.open(path)
        image = image.resize((150, 150))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def display_prediction(self):
        prediction = make_prediction(self.image_path)
        self.display_image(f"dataset/Meta/{prediction}.png", self.output_label)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x400")  # Adjusted window size
    app = ImageDisplayApp(root)
    root.mainloop()
