import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import os


def flip_horizontal(image):
    flipped = [[0 for _ in range(image.shape[1])]
               for _ in range(image.shape[0])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flipped[i][j] = image[i][image.shape[1] - 1 - j]
    return flipped


def flip_vertical(image):
    flipped = [[0 for _ in range(image.shape[1])]
               for _ in range(image.shape[0])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flipped[i][j] = image[image.shape[0] - 1 - i][j]
    return flipped


def rotate_90(image):
    rotated = [[0 for _ in range(image.shape[0])]
               for _ in range(image.shape[1])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rotated[j][image.shape[0] - 1 - i] = image[i][j]
    return rotated


def rotate_minus_90(image):
    rotated = [[0 for _ in range(image.shape[0])]
               for _ in range(image.shape[1])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rotated[image.shape[1] - 1 - j][i] = image[i][j]
    return rotated


def resize(image, new_shape):
    resized = [[0 for _ in range(new_shape[1])] for _ in range(new_shape[0])]
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            x = int(i * (image.shape[0] / new_shape[0]))
            y = int(j * (image.shape[1] / new_shape[1]))
            resized[i][j] = image[x][y]
    return resized


def create_download_link(image, filename):
    temp_file = f"{filename}.temp.jpg"
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(temp_file, rgb_image)
    with open(temp_file, "rb") as f:
        image_bytes = f.read()
    os.remove(temp_file)
    b64 = base64.b64encode(image_bytes).decode()
    button_html = f'<button style="background-color: blue;"><a href="data:image/png;base64,{b64}" download="{filename}" style="color: white; text-decoration: none;">Save to your PC</a></button>'
    return button_html


def main():
    st.title("Image Transformation App without OpenCV and Numpy")
    st.text(
        "Here is the code to implement image transformation without OpenCV and Numpy")
    st.code("""
def flip_horizontal(image):
    flipped = [[0 for _ in range(image.shape[1])]
               for _ in range(image.shape[0])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flipped[i][j] = image[i][image.shape[1] - 1 - j]
    return flipped


def flip_vertical(image):
    flipped = [[0 for _ in range(image.shape[1])]
               for _ in range(image.shape[0])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flipped[i][j] = image[image.shape[0] - 1 - i][j]
    return flipped


def rotate_90(image):
    rotated = [[0 for _ in range(image.shape[0])]
               for _ in range(image.shape[1])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rotated[j][image.shape[0] - 1 - i] = image[i][j]
    return rotated


def rotate_minus_90(image):
    rotated = [[0 for _ in range(image.shape[0])]
               for _ in range(image.shape[1])]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rotated[image.shape[1] - 1 - j][i] = image[i][j]
    return rotated


def resize(image, new_shape):
    resized = [[0 for _ in range(new_shape[1])] for _ in range(new_shape[0])]
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            x = int(i * (image.shape[0] / new_shape[0]))
            y = int(j * (image.shape[1] / new_shape[1]))
            resized[i][j] = image[x][y]
    return resized""", language="python")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = np.array(image)

        transformation = st.selectbox("Choose a transformation", [
                                      "Flip Horizontal", "Flip Vertical", "Rotate 90", "Rotate -90", "Resize"])

        if st.button("Apply Transformation"):
            if transformation == "Flip Horizontal":
                transformed_image = flip_horizontal(image)
            elif transformation == "Flip Vertical":
                transformed_image = flip_vertical(image)
            elif transformation == "Rotate 90":
                transformed_image = rotate_90(image)
            elif transformation == "Rotate -90":
                transformed_image = rotate_minus_90(image)
            elif transformation == "Resize":
                transformed_image = resize(image, (800, 450))

            transformed_image = np.array(transformed_image, dtype=np.uint8)
            st.image(transformed_image, caption="Transformed Image",
                     use_column_width=True)

            download_link = create_download_link(
                transformed_image, f"{transformation.replace(' ', '_')}.jpg")
            st.markdown(download_link, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
