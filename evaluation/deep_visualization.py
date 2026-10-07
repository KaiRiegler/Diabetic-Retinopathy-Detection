import gin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import wandb


@gin.configurable
class Grad_CAM:
    '''
    A class to compute and visualize Grad-CAM. Grad-CAM highlights the regions of an image
    that are most relevant for a specific prediction.

        Attributes:
                model (tf.keras.Model): The neural network to apply Grad-CAM on.
                class_idx (int): The index of the target class for visualization. 
                                    If None, the class with the highest prediction score is used.
                last_conv_layer_name (str): The name of the last convolutional layer in the model.
                                            If None, it is determined dynamically.
    
    '''
    def __init__(self, model, class_idx, last_conv_layer_name):
        self.model = model
        self.class_idx = class_idx
        self.last_conv_layer_name= last_conv_layer_name

        # Check if model has functional layer
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.Model):
                # unwrapp model
                input = self.model.get_layer(layer.name).input
                output = self.model.get_layer(layer.name).output
                for i in range(idx+1, len(self.model.layers)):
                    output = self.model.get_layer(index=i)(output)
                self.model = tf.keras.Model(input, output)

        if self.last_conv_layer_name is None:
            self.last_conv_layer_name = self.find_last_conv_layer()

    
    def find_last_conv_layer(self):
        '''
        Find the last convolutional layer in the model.

            Returns:
                    self (str): Name of the last convolutional layer in the model.

            Raises:
                    ValueError: If no convolutional layer is found in the model.
        '''
        for layer in reversed(self.model.layers):
            # Check last conv layer of model in reversed order
            if isinstance(layer, tf.keras.layers.Conv2D):
                # print(f"Last Conv2D Layer: {layer.name}")
                return layer.name

        raise ValueError("Could not find Conv layer. Cannot apply GradCAM.")
    

    def compute_heatmap(self, image_tensor):
        '''
        Computes the Grad-CAM heatmap for a given image.

            Parameters:
                    image_tensor (tf.Tensor): The input image tensor with shape (1, height, width, channels).

            Returns:
                    heatmap (np.ndarray): The Grad-CAM heatmap as a numpy array.
                    class_idx (int): Predicted label of the image_tensor.
                    confidence (float): Model's confidence score for the predicted label.
        '''
        # Create a model that outputs the activations of the last conv layer and predictions
        grad_model = tf.keras.models.Model([self.model.inputs], 
                                          [self.model.get_layer(self.last_conv_layer_name).output, self.model.output])
        
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)

            if self.class_idx is None:
                self.class_idx = tf.argmax(predictions[0])
            confidence = predictions[0][self.class_idx].numpy()  # Extract confidence for predicted label
            loss = predictions[:, self.class_idx]

        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)

        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute the headmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.squeeze(heatmap) # Remove single-dimensional entries

        # Normalize heatmap to range [0, 1]
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy(), self.class_idx.numpy(), confidence
    

    @gin.configurable
    def overlay_and_display_heatmap(self, image_tensor, heatmap, true_label, predicted_label, confidence, alpha, colormap, mode_norm):
        '''
        Overlays the Grad-CAM heatmap on the original image and displays the results.

            Parameters: 
                    image_tensor (tf.Tensor): The original input image tensor.
                    heatmap (np.ndarray): The Grad-CAM heatmap.
                    true_label (int): True label of an image.
                    predicted_label (int): Predicted label of an image.
                    confidence (float): Model's confidence score for the predicted label.
                    alpha (float): Transparency level for the heatmap overlay.
                    colormap (str): The colormap to use for the heatmap.
                    mode_norm (bool): choose normalization mode (True: [0,1], False: [-1,1]).
        '''
        # Scale heatmap to 0-255 range and convert to uint8
        heatmap = np.uint8(255 * heatmap) 
        
        # Create a colormap and map the heatmap values to RGB
        cmap = cm.get_cmap(colormap)
        cmap_colors = cmap(np.arange(256))[:, :3]
        cmap_heatmap = cmap_colors[heatmap]

        # Convert heatmap to an image and resize to match input image size
        cmap_heatmap = tf.keras.utils.array_to_img(cmap_heatmap)
        cmap_heatmap = cmap_heatmap.resize((image_tensor.shape[0], image_tensor.shape[1]))
        cmap_heatmap = tf.keras.utils.img_to_array(cmap_heatmap)

        # Overlay heatmap on the original image with transparency
        if not mode_norm:
            image_tensor = (image_tensor + 1.0) / 2.0 # from [-1,1] to [0,1]
        image_tensor = image_tensor * (1 - alpha)
        image_tensor = tf.image.convert_image_dtype(image_tensor, tf.uint8) # Change image to unit8 for overlay
        overlayed_image = cmap_heatmap * alpha + image_tensor

        # Plot the results: heatmap, original image, and overlayed image
        # Subplot Heatmap
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Grad-Cam Heatmap")
        plt.imshow(heatmap, cmap=colormap)
        plt.axis("off")

        # Subplot Original Image
        plt.subplot(1, 3, 2)
        plt.title("Original Image")
        plt.imshow(image_tensor)
        plt.axis("off")

        # Subplot Heatmap Overlay
        plt.subplot(1, 3, 3)
        plt.title(f"Grad-CAM Heatmap Overlay\nTrue Label: {true_label}, Pred Label: {predicted_label}")
        plt.imshow(overlayed_image)
        plt.axis("off")

        # Add confidence to the bottom-left corner
        plt.text(
            x=4.0, y=16.0,  # Position in axes coordinates (0, 0 = bottom-left)
            s=f"Confidence: {confidence * 100:.2f}%",  # Display confidence as percentage
            color="white", fontsize=9, ha="left", va="bottom",
            bbox=dict(facecolor="black", alpha=0.6))  # Background styling
        
        plt.tight_layout()
        wandb.log({"Some Grad-Cam plots": wandb.Image(plt)})
        plt.close()


@gin.configurable
def generate_grad_cam_heatmap(model, ds_test, num_batches_displayed, num_items_displayed):
    '''
    Creates an Grad-CAM object and displays Grad-CAM heatmaps for a given model and test dataset.

        Parameters:
                model (tf.keras.Model): The trained model for which Grad-CAM heatmaps are to be generated.
                ds_test (tf.data.Dataset): The test dataset containing image-label pairs.
                num_batches_displayed (int): The number of batches to process from the test dataset.
                num_items_displayed (int): The number of images per batch to display Grad-CAM results for.
        
        Raises:
                ValueError: If `num_items_displayed` is greater than the number of items in the batch.
    '''
    # print("Generating Crad-Cam visualization...")
    cam = Grad_CAM(model) # Create Grad-CAM object
    for batch_images, batch_labels in ds_test.take(num_batches_displayed): # Takes i-th batches
        
        # Check if num_items_displayed exceeds the batch size
        batch_size = tf.shape(batch_images)[0]
        if num_items_displayed > batch_size:
            raise ValueError(f"`num_items_displayed` ({num_items_displayed}) exceeds the number of items in the batch ({batch_size}).")

        for i in range(num_items_displayed): # Displays first i-th samples of an batch
            image = batch_images[i]
            true_label = batch_labels[i]
            image_for_model = tf.expand_dims(image, axis=0)

            heatmap, predicted_label, confidence = cam.compute_heatmap(image_for_model)
            cam.overlay_and_display_heatmap(image, heatmap, true_label, predicted_label, confidence)
    # print("Finished Crad-Cam visualization!")
