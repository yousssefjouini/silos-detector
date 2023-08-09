from keras.models import load_model
import importlib.resources
import os
from keras.utils import load_img, img_to_array
import pandas as pd
from typing import Any
from PIL import Image
import numpy as np


class SilosClassification:
    """Loads a CNN classification model to predict on files or folders.

    Method :

    predict(filepath) : given a filepath, returns either a float if the filepath is a file or a pd.DataFrame with filenames and
    associated predictions if the filepath is a folder.
    """

    def __init__(self) -> None:
        with importlib.resources.path(
            "silos.classification.prediction.models", "cnn_nath.hdf5"
        ) as model_path:
            self.model = load_model(model_path, compile=False)

    def predict(self, to_predict: str, from_buffer: bool = False) -> Any:
        if from_buffer:
            uploaded_file = Image.open(to_predict)
            uploaded_file = uploaded_file.resize((32, 32))
            uploaded_file = np.array(uploaded_file)
            uploaded_file = uploaded_file / 255
            uploaded_file = uploaded_file[None, :]
            prediction = self.model.predict(uploaded_file)
            return prediction[0][0]

        elif os.path.isfile(to_predict):
            img = load_img(to_predict, target_size=(32, 32, 1), grayscale=False)
            img = img_to_array(img)
            img = img / 255
            img = img[None, :]
            prediction = self.model.predict(img)
            return prediction[0][0]

        elif os.path.isdir(to_predict):
            results = []
            for f in [
                file
                for file in os.listdir(to_predict)
                if os.path.isfile(os.path.join(to_predict, file))
            ]:
                img = load_img(
                    os.path.join(to_predict, f),
                    target_size=(32, 32, 1),
                    grayscale=False,
                )
                img = img_to_array(img)
                img = img / 255
                img = img[None, :]
                prediction = self.model.predict(img)
                results.append((f, prediction[0][0]))
                results_df = pd.DataFrame(results, columns=["name", "prediction"])
            return results_df
