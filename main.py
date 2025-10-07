from data import get_data_generators
from model import build_transfer_learning_model
from train import train_model
from evaluate import evaluate_model
from plot import plot_training_history, plot_confusion_matrix
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, BATCH_SIZE
import tensorflow as tf

# Data prep
train_gen, val_gen, test_gen = get_data_generators()
class_names = list(train_gen.class_indices.keys())

# Model
model = build_transfer_learning_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = train_model(model, train_gen, val_gen)

# Evaluate
cm = evaluate_model(model, test_gen, class_names)

# Plot
plot_training_history(history)
plot_confusion_matrix(cm, class_names)

# Save
model.save("intel_image_classification_final_model.keras")
