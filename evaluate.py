from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_generator, class_names):
    test_steps = (test_generator.samples + test_generator.batch_size - 1) // test_generator.batch_size
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    y_pred = model.predict(test_generator, steps=test_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred_classes)
    return cm
