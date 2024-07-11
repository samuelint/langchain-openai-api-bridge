import base64
import os


def base64_pig_image():
    image_path = os.path.join(os.path.dirname(__file__), "../assets/pig.jpg")
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        return base64.b64encode(img_bytes).decode()


def base64_url_pig_image():
    return f"data:image/jpeg;base64,{base64_pig_image()}"
