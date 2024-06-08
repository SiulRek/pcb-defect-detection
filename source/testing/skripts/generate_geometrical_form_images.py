import os
import random

from PIL import Image, ImageDraw


def generate_geometrical_form_image(
    path, width, height, image_number, image_format="png"
):
    """
    Generate an image with noise, a rectangle, and a circle, and save it to the
    specified path.

    Args:
        - path (str): Directory path to save the generated image.
        - width (int): Width of the image.
        - height (int): Height of the image.
        - image_number (int): The image number to be used in the filename.
        - image_format (str): The format to save the image. Defaults to
            "png".
    """
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for x in range(width):
        for y in range(height):
            if random.random() < 0.80:  # 80% of the pixels will have noise
                noise_color = (
                    random.randint(150, 180),
                    random.randint(100, 120),
                    random.randint(60, 100),
                )
                draw.point((x, y), fill=noise_color)

    rect_x0, rect_y0 = width // 4, height // 4
    rect_x1, rect_y1 = 3 * width // 4, 3 * height // 4
    rect_color = (80, 180, 120)
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], outline=rect_color, width=10)

    circle_x0, circle_y0 = width // 3, height // 3
    circle_x1, circle_y1 = 2 * width // 3, 2 * height // 3
    circle_color = (120, 80, 180)
    draw.ellipse(
        [circle_x0, circle_y0, circle_x1, circle_y1], outline=circle_color, width=10
    )

    image.save(os.path.join(path, f"image_{image_number}.{image_format}"))


def generate_geometrical_form_images(
    path, num_images, width=600, height=400, start_number=1, image_format="png"
):
    """
    Generate multiple images of geometrical forms and save them to the specified
    path.

    Args:
        - path (str): Directory path to save the generated images.
        - num_images (int): Number of images to generate.
        - width (int, optional): Width of the images. Defaults to 600.
        - height (int, optional): Height of the images. Defaults to 400.
        - start_number (int, optional): Starting number for image filenames.
            Defaults to 1.
        - image_format (str, optional): The format to save the images.
            Defaults to "png".
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(start_number, start_number + num_images):
        generate_geometrical_form_image(path, width, height, i, image_format)


if __name__ == "__main__":
    output_path = "./source/testing/image_data/geometrical_forms"
    number_of_images = 5  # Number of images to generate
    start_number = 1  # Starting number for image filenames
    image_format = "png"  # Image format
    width = 600  # Width of the images
    height = 400  # Height of the images

    generate_geometrical_form_images(
        output_path,
        number_of_images,
        width=width,
        height=height,
        start_number=start_number,
        image_format=image_format,
    )
