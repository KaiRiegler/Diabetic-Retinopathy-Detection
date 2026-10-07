import gin
import tensorflow as tf


@gin.configurable
def preprocess(image, label, img_height, img_width, offset_height, offset_width, bounding_box_height, bounding_box_width, mode_norm):
    '''
    Preprocesses the images by normalizing, cropping and resizing with padding.

        Parameters:
                image (tf.Tensor): Input image tensor.
                label (int): The label corresponding to the image.
                img_height (int): The target height of the resized image.
                imag_width (int): The target width of the resized image.
                offset_height (int): Vertical coordinate of the top-left corner of the bounding box in image.
                offset_width (int): Horizontal coordinate of the top-left corner of the bounding box in image.
                bounding_box_height (int): Height of the bounding box.
                bounding_box_width (int): Width of the bounding box.
                mode_norm (bool): choose normalization mode (True: [0,1], False: [-1,1]).
    
        Returns:
                preprocessed_image (tf.Tensor): Preprocessed image tensor.
                label (int): Corresponding label to the image tensor.
    '''
    # Normalize image: `uint8` -> `float32`.
    image = tf.image.convert_image_dtype(image, tf.float32) # [0,1]

    # Crop image
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, bounding_box_height, bounding_box_width)
    
    # Resize image and add padding 
    preprocessed_image = tf.image.resize_with_pad(cropped_image, target_height=img_height, target_width=img_width)

    if not mode_norm:
        preprocessed_image = preprocessed_image * 2 - 1 # Normalize to [-1,1], important for transfer learning 

    return preprocessed_image, label


def load_image(file_path, label):
    '''
    Load a single image.

        Parameters:
                file_path (str): Path to the image file.
                label (int): The label corresponding to the image.

        Returns:
                image (tf.Tensor): Loaded image tensor.
                label (int): Corresponding label to the image tensor.
    '''
    # Read image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, label


def augment(image, label):
    '''
    Apply random augmentation to a single image.

        Parameters:
                image (tf.Tensor): Input image tensor.
                label (int): The label corresponding to the image.

        Returns:
                image (tf.Tensor): Augmented image tensor.
                label (int): Corresponding label to the image tensor.
    '''
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label
