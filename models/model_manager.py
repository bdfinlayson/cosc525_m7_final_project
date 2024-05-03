from __future__ import annotations

import io
from typing import Type
from io import StringIO
import sys

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
import numpy as np
from keras_tuner import RandomSearch
from keras.models import load_model
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib as mpl

from data.data_manager import DataManager
from models.model_1 import Model1
from models.model_2 import Model2
from constants.klass_mappings import klass_mappings


class ModelManager:
    model_to_init: Type[Model1, Model2]
    hypermodel: kt.HyperModel
    data_manager: DataManager
    tuner: kt.RandomSearch
    best_model: any
    best_hyperparameters: dict
    model_filepath: str
    model_name: str
    tuning_epochs: int
    training_epochs: int
    history: keras.callbacks.History
    num_classes: int
    input_shape: int
    desired_shape: tuple[int, int, int]
    max_trials: int

    def __init__(self, model_to_init: Type[Model1, Model2], model_name: str, max_trials=10, training_epochs=10, tuning_epochs=5, use_preprocessed_data=False):
        self.model_to_init = model_to_init
        self.data_manager = DataManager()
        self.model_name = f'{model_name}_24x24' if use_preprocessed_data else model_name
        self.model_filepath = f'./models/snapshots/{self.model_name}.keras'
        self.tuning_epochs = tuning_epochs
        self.training_epochs = training_epochs
        self.num_classes = len(klass_mappings)
        self.input_shape = (4096, 1)
        self.desired_shape = (64, 64, 1)
        self.max_trials = max_trials

    def run_pipeline(self):
        self.build_model()\
            .prepare_data()\
            .search()\
            .set_best_model()\
            .plot_model_diagram()\
            .set_best_model_hyperparameters()\
            .compile_model()\
            .fit_model()\
            .generate_loss_and_accuracy_chart() \
            .generate_model_metrics_chart() \
            .generate_confusion_matrix_chart() \
            .generate_model_summary()\
            .generate_best_hyperparams_chart()\
            .test_model()

    def build_model(self) -> ModelManager:
        self.hypermodel = self.model_to_init(input_shape=self.input_shape, desired_shape=self.desired_shape, num_classes=self.num_classes)
        return self

    def prepare_data(self) -> ModelManager:
        self.data_manager.load_data()
        return self

    def search(self) -> ModelManager:
        self.tuner = RandomSearch(self.hypermodel,
                                  objective="val_accuracy",
                                  max_trials=self.max_trials,
                                  max_consecutive_failed_trials=10,
                                  executions_per_trial=1,
                                  directory='tuning',
                                  project_name="hyperparams",
                                  overwrite=True)
        self.tuner.search(self.data_manager.X_train_sample, self.data_manager.y_train_sample, epochs=self.tuning_epochs, validation_data=(self.data_manager.X_train_sample, self.data_manager.y_train_sample))
        return self

    def set_best_model(self) -> ModelManager:
        self.best_model = self.tuner.get_best_models(num_models=1)[0]
        return self

    def plot_model_diagram(self) -> ModelManager:
        tf.keras.utils.plot_model(self.best_model, to_file=f'./images/{self.model_name}_diagram.png', show_shapes=True)
        return self

    def set_best_model_hyperparameters(self) -> ModelManager:
        best_hyperparams = self.tuner.get_best_hyperparameters(10)[0]
        self.best_hyperparameters = best_hyperparams.values
        print("Best Hyperparameters:\n", self.best_hyperparameters)
        return self

    def compile_model(self) -> ModelManager:
        self.best_model.compile(loss="binary_crossentropy",
                                optimizer="rmsprop",
                                metrics=['accuracy', 'precision', 'recall', 'AUC'])
        return self

    def fit_model(self) -> ModelManager:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.model_filepath,
                save_best_only=True,
                monitor="val_loss")
        ]
        self.history = self.best_model.fit(
            x=self.data_manager.X_train,
            y=self.data_manager.y_train,
            epochs=self.training_epochs,
            validation_data=(self.data_manager.X_val, self.data_manager.y_val),
            callbacks=callbacks)
        return self

    def generate_loss_and_accuracy_chart(self) -> ModelManager:
        accuracy = self.history.history["accuracy"]
        val_accuracy = self.history.history["val_accuracy"]
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, "bo", label=f"Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title(f"{self.model_name}\n\nTraining and validation accuracy")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.savefig(f"./images/{self.model_name}_loss_and_accuracy.png")
        plt.show()
        return self

    def generate_model_metrics_chart(self) -> ModelManager:
        plt.plot(self.history.history['accuracy'], label='Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(self.history.history['precision'], label='Precision')
        plt.plot(self.history.history['recall'], label='Recall')
        plt.plot(self.history.history['AUC'], label='AUC')
        plt.title(f'{self.model_name.title()} Model Metrics')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(loc="upper left")
        plt.savefig(f"./images/{self.model_name}_model_metrics.png")
        plt.show()
        return self

    def generate_confusion_matrix_chart(self) -> ModelManager:
        model = load_model(self.model_filepath)
        predictions = model.predict(self.data_manager.X_test)
        predicted_labels = np.argmax(predictions, axis=1) if predictions.ndim > 1 else (predictions > 0.5).astype(int)
        actual_labels = np.argmax(self.data_manager.y_test, axis=1)
        cm = confusion_matrix(actual_labels, predicted_labels)
        plt.figure(figsize=(30, 25))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', norm=LogNorm(), linewidths=0.5, linecolor='black', annot_kws={"size": 11})
        plt.xlabel('Predicted Labels', fontsize=30)
        plt.ylabel('True Labels', fontsize=30)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Count', size=30)
        plt.title(f'{self.model_name.title()} Confusion Matrix', fontsize=40)
        plt.savefig(f"./images/{self.model_name}_confusion_matrix.png")
        plt.show()
        return self

    def test_model(self) -> ModelManager:
        test_model = keras.models.load_model(self.model_filepath)
        loss, accuracy, precision, recall, auc = test_model.evaluate(x=self.data_manager.X_test, y=self.data_manager.y_test)

        # Organize data into a DataFrame
        data = {
            "Metric": ["Accuracy", "Loss", "Precision", "Recall", "AUC"],
            "Value": [
                f"{accuracy:.3f}",
                f"{loss:.3f}",
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{auc:.3f}"
            ]
        }
        df = pd.DataFrame(data)

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('tight')
        ax.axis('off')
        fig.patch.set_facecolor('#f4f4f4')
        fig.suptitle(f'{self.model_name.title()} Test Metrics', fontsize=16, y=0.95)
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                         colColours=["#40466e"]*2, cellColours=[["#f4f4f4"]*2]*len(df))
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        # Modify padding of the cells
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_height(0.15)
            cell.set_linewidth(1.2)
            if key[0] == 0:
                cell.set_fontsize(14)
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
        plt.tight_layout()
        plt.savefig(f"./images/{self.model_name}_test_results.png", dpi=300)
        plt.show()
        plt.close(fig)
        return self

    def generate_model_summary(self) -> ModelManager:
        s = io.StringIO()
        self.best_model.summary(print_fn=lambda x: s.write(x + '\n'))
        model_summary = s.getvalue()
        s.close()

        # print for good measure
        self.best_model.summary(print_fn=lambda x: print(x))

        # Redirect stdout to a StringIO object to capture the output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        print(model_summary)
        # Get the output and reset sys.stdout
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        def text_to_image(text, filename='output.png'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'{self.model_name.title()} Model Summary', y=1.10, fontsize=14)
            ax.text(0.5, 0.5, text, fontsize=12, va='center', ha='center', transform=ax.transAxes)
            ax.axis('off')  # Hide the axes
            ax.axis('tight')
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

        text_to_image(output, f'./images/{self.model_name.title()}_model_summary.png')
        return self

    def generate_best_hyperparams_chart(self) -> ModelManager:
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(list(self.best_hyperparameters.items()), columns=['Parameter', 'Value'])

        # Set the matplotlib style
        mpl.style.use('grayscale')

        # Create a plot with specific figure size
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')

        # Create a table in the plot
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.5)

        # Style the table for better aesthetics
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)
            if key[0] == 0:
                cell.set_fontsize(16)
                cell.set_facecolor('darkgray')
                cell.set_text_props(color='white')

        # Adjust layout to make sure everything fits and looks nice
        plt.tight_layout()

        plt.title(f'{self.model_name.title()} Best Hyperparameters', fontsize=18, y=1.10)

        # Save the figure to an image
        plt.savefig(f'./images/{self.model_name}_best_hyperparameters.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return self
