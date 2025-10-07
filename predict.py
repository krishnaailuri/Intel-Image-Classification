from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, image_path, img_height, img_width, class_names):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]
    print(f"Predicted class: {pred_class}")
