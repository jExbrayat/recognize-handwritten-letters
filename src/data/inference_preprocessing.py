import cv2
import numpy as np
from PIL import Image, ImageOps


def crop_image(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    return img.crop(bbox)


def pad_image(img: Image.Image, padding: int = 4) -> Image.Image:
    new_width = img.width + 2 * padding
    new_height = img.height + 2 * padding

    img_padded = Image.new("L", (new_width, new_height), 0)
    img_padded.paste(img, (padding, padding))

    return img_padded


def turn_to_mnist(img: Image.Image) -> Image.Image:
    img_inverted = ImageOps.invert(img)
    img_cropped = crop_image(img_inverted)
    img_resized = img_cropped.resize((28, 28), Image.Resampling.LANCZOS)
    img_padded = pad_image(img_resized)
    img_resized = img_padded.resize((28, 28), Image.Resampling.LANCZOS)
    img_dilated = dilatate(img_resized)
    img_more_contrasted = intensify_contrast(img_dilated)

    return img_more_contrasted


def dilatate(img: Image.Image) -> Image.Image:
    kernel = np.ones((2, 2), np.uint8)

    dilated = cv2.dilate(np.array(img), kernel, iterations=1)
    img_dilated = Image.fromarray(dilated)
    return img_dilated


def intensify_contrast(img: Image.Image) -> Image.Image:
    """Enhance the contrast of an image by stretching its intensity range.

    This function uses Pillow's ``ImageOps.autocontrast`` with a cutoff of 5%.
    It removes the darkest 5% and brightest 5% of pixel values and rescales the
    remaining range to [0, 255], making black darker and white brighter while
    keeping intermediate grays.

    Parameters
    ----------
    img : PIL.Image.Image
        Input grayscale image.

    Returns
    -------
    PIL.Image.Image
        Image with intensified contrast.

    Examples #todo
    --------
    >>> img = Image.fromarray(np.array([[0, 255], [100, 200]], dtype=np.uint8))
    >>> result = intensify_contrast(img)
    >>> isinstance(result, Image.Image)
    True
    """
    return ImageOps.autocontrast(img, cutoff=5)
