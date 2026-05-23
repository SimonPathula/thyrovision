import numpy as np
from utils.logger import logger

def preprocess_img(image):
    logger.info(f"Preprocessing image of size: {image.size} and {image.mode}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224))
    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis= 0)
    img_array = img_array.astype("float32") /255.0

    return img_array