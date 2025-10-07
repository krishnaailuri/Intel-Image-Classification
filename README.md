# Intel Image Classification

This project implements an image classification system for Intel's image dataset, classifying images into six categories: buildings, forest, glacier, mountain, sea, and street. It leverages convolutional neural networks and transfer learning with VGG16 to achieve accurate classification results. Additionally, it includes a Flask-based web app for uploading images and getting predictions.

## Project Structure

- `app.py`: Flask application for image upload and prediction.
- `config.py`: Configuration constants for dataset paths, image size, batch size, epochs, and classes.
- `data.py`: Data generators for training, validation, and testing using Keras `ImageDataGenerator`.
- `model.py`: Custom CNN and transfer learning model building functions.
- `train.py`: Model training including callbacks for early stopping and best model saving.
- `evaluate.py`: Model evaluation producing accuracy, classification report, and confusion matrix.
- `plot.py`: Visualization utilities for training history and confusion matrix.
- `predict.py`: Utility for predicting the class of a single image.
- `main.py`: Script orchestrating the training, evaluation, plotting, and model saving.


archive/
├── seg_pred/         # Predicted segmentation outputs
├── seg_test/         # Test dataset for evaluation
└── seg_train/        # Training dataset
    ├── buildings/    # Training images/masks for buildings class
    ├── forest/       # Training images/masks for forest class
    ├── glacier/      # Training images/masks for glacier class
    ├── mountain/     # Training images/masks for mountain class
    ├── sea/          # Training images/masks for sea class
    └── street/       # Training images/masks for street class

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create and activate a Python virtual environment (optional but recommended):
   ```
    source tf-env/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare the dataset as per the directory structure specified in `config.py`.

## Usage

### Training the Model
Run the main script which handles data loading, training, evaluation, plotting, and saving:
```
python3 -X faulthandler main.py

```

### Running the Flask App
To use the web app, ensure you have the best saved model (`intel_image_classification_best_model.keras`), then run:
```
python3 -X faulthandler app.py

```
Open the browser and navigate to `http://127.0.0.1:5000` to upload images for prediction.

## Features

- Data augmentation and preprocessing with Keras.
- Custom CNN and transfer learning with VGG16.
- Model training with early stopping and checkpointing.
- Comprehensive model evaluation with classification reports and confusion matrix.
- Visualizations for training progress and performance.
- Flask web app for easy image classification.

## Contributing

Feel free to fork the repository, improve the model, and submit pull requests. For major changes, please open an issue first.

## License

This project is licensed under the MIT License.
```

```
Flask>=2.0.0
tensorflow>=2.6.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```
