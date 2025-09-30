from keras.applications.efficientnet import preprocess_input, decode_predictions
import cv2
import numpy as np
import tensorflow as tf




import tensorflow as tf
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
import numpy as np

# Load the model
model = tf.keras.models.load_model('models\efficientnetb0-Indian Rock Python-98.80.h5',compile=False)

# Load the weights
model.load_weights('models\efficientnetb0-Indian Rock Python-weights.h5')

# Load the test image (replace 'path_to_test_image.jpg' with the actual path)
def pred(image_path):
    print(image_path)
    test_image_path=image_path
    img = image.load_img(test_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Interpret predictions based on your model's output
    # Assuming the model outputs two values: [probability_not_pneumonia, probability_pneumonia]
    probability_not_pneumonia, probability_pneumonia = predictions[0]
    return probability_pneumonia,probability_not_pneumonia

    

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

def custom_segmentation(image_path):
    # Load and preprocess the X-ray image
    mask_image_path="/test images/Normal_segment.jpeg"
    img = image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]

    # Load the mask image
    mask_img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask image to match the X-ray image size
    mask_img = cv2.resize(mask_img, (224, 224))

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(img_array[0, :, :, 0], (5, 5), 0)

    # Convert the floating-point image to 8-bit unsigned integer
    blurred_image_uint8 = cv2.convertScaleAbs(blurred_image)

    # Apply adaptive thresholding to create a segmentation mask
    _, segmentation_mask = cv2.threshold(blurred_image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure that both images have the same data type
    img_array_bgr = cv2.cvtColor(cv2.merge([img_array[0], img_array[0], img_array[0]]), cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Overlay the segmentation mask on the original image
    overlay_segmentation = cv2.addWeighted(img_array_bgr, 0.5,
                                          cv2.cvtColor(cv2.merge([segmentation_mask, segmentation_mask, segmentation_mask]), cv2.COLOR_BGR2RGB).astype(np.uint8), 0.5, 0)

    # Ensure that both images have the same data type
    mask_img_bgr = cv2.cvtColor(cv2.merge([mask_img, mask_img, mask_img]), cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Overlay the mask image on the original image
    overlay_mask = cv2.addWeighted(overlay_segmentation, 1.0, mask_img_bgr, 0.5, 0)

    # Visualize the original image with the segmentation mask and mask overlay
    plt.imshow(overlay_mask)
    plt.title('Original Image with Segmentation Mask and Mask Overlay')
    plt.show()

    # Calculate severity based on the number of opacity pixels (this is a simplified example)
    total_opacity_pixels = np.sum(segmentation_mask == 255)
    severity_percentage = total_opacity_pixels / (224 * 224) * 100.0

    return severity_percentage


