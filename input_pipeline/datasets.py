import os
import gin
import logging
import gin.config
import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd
from sklearn.model_selection import train_test_split

from input_pipeline.preprocessing import preprocess, load_image, augment


@gin.configurable
def load(name,
         data_dir,
         use_binary_labels_idrid,
         original_label_column,
         binary_labels_column,
         data_dir_train_images,
         data_dir_train_labels,
         data_dir_test_images,
         data_dir_test_labels,
         val_split):
    '''
    Load the dataset, create binary labels for IDRID dataset, splits it into training, validation and test set and 
    returns TensorFlow datasets. 

        Parameters:
                name (str): Name of the dataset.
                data_dir (str): Path to directory containing the datasets.
                use_binary_labels_idrid (bool): Whether to create and use binary labels for IDRID dataset.
                original_label_column (str): Name of the column with original labels.
                binary_labels_column (str): Name of the column with binary labels.
                data_dir_train_images (str): Path to train images inside IDRID dataset directory.
                data_dir_train_labels (str): Path to train labels inside IDRID dataset directory.
                data_dir_test_images (str): Path to test images inside IDRID dataset directory.
                data_dir_test_labels (str): Path to test labels inside IDRID dataset directory.
                val_split (int): Ration to split train data into training and validation sets.
        
        Returns:
                Tuple [tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Optional[dict]]:
                    A tuple containing:
                        - 'ds_train': Training dataset
                        - 'ds_val': Validation dataset
                        - 'ds_test': Testing dataset
                        - 'ds_info': Optional metadata of information about the dataset
    '''
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")   

        # Read CSV files
        train_df = pd.read_csv(os.path.join(data_dir, data_dir_train_labels))
        test_df = pd.read_csv(os.path.join(data_dir, data_dir_test_labels))

        label_column_name = original_label_column

        # Create binary labels for 2-class classification 
        if use_binary_labels_idrid:
            train_df = create_binary_labels_idrid(train_df, original_label_column, binary_labels_column)
            test_df = create_binary_labels_idrid(test_df, original_label_column, binary_labels_column)
            label_column_name = binary_labels_column

        # Split train into train and validation
        train_df, val_df = train_test_split(train_df, test_size=val_split, random_state=42, stratify=train_df[label_column_name])

        # Create tf.data.Dataset from image directory and labels
        ds_train = create_dataset(os.path.join(data_dir, data_dir_train_images), train_df, label_column_name)
        ds_val = create_dataset(os.path.join(data_dir, data_dir_train_images), val_df, label_column_name)
        ds_test = create_dataset(os.path.join(data_dir, data_dir_test_images), test_df, label_column_name)
 
        return prepare(ds_train, ds_val, ds_test, ds_info=None)

    else:
        raise ValueError



@gin.configurable
def create_dataset(images_dir, labels_df, used_labels, image_id):
    '''
    Create a Tenserflow dataset from a directory of images and a DataFrame of labels.

        Parameters:
                images_dir (str): Path to the directory containing image files.
                labels_df (pd.DataFrame): A DataFrame containing image id and label columns.
                image_id (str): Name of the column with the image id.
                used_labels (str): Name of the column of the used labels - original or binary labels.

        Returns:
                dataset (tf.data.Dataset): A Tenserflow dataset with loaded images.
    '''
    # Create tf.data.Dataset from image directory and labels.
    file_paths = labels_df[image_id].apply(lambda x: os.path.join(images_dir, f"{x}.jpg")).tolist()
    labels = labels_df[used_labels].tolist()
 
    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Load images to dataset
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset



def create_binary_labels_idrid(df, source_column, target_column):
    '''
    Map multi-class labels to binary labels.
    
        Parameters:
                df (pd.DataFrame): DataFrame containing the labels.
                source_column (str): Name of the column with original labels.
                target_column (str): Name of the column to store binary labels.

        Returns:
                df (pf.DataFrame): Updated DataFrame with binary labels.
    '''
    # Example mapping logic: 0 or 1 -> 0, 2-4 -> 1
    df[target_column] = df[source_column].apply(lambda label: 0 if (label == 0 or label == 1) else 1)

    return df



@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, buffer_size, caching):
    '''
    Prepares Tensorflow datasets for training, validation and testing by applying
    batching, shuffling, caching, prefetching and augmentation.

        Parameters:
                ds_train (tf.data.Dataset): Training dataset.
                ds_val (tf.data.Dataset): Validation dataset.
                ds_test (tf.data.Dataset): Test dataset.
                ds_info (tfds.core.DatasetInfo): Optional metadata of information about the dataset.
                batch_size (int): Size of each batch.
                buffer_size (int): Buffer size for shuffling the training data.
                caching (bool): Whether to cache the datasets for improved performance.
        
        Returns:
                ds_train (tf.data.Dataset): Prepared training dataset.
                ds_val (tf.data.Dataset): Prepared validation dataset.
                ds_test (tf.data.Dataset): Prepared test dataset.
                ds_info(tfds.core.DatasetInfo): Optional metadata of information about the dataset.
    '''
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
