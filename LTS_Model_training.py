"""
Liver Tumor Segmentation Model Training Module

This module provides functionality for training a deep learning model for liver tumor
segmentation using CT scan data. It processes NIfTI (.nii) files, applies DICOM windowing
techniques, and trains a U-Net model using the FastAI framework.

Key Features:
- NIfTI file processing and data preparation
- DICOM windowing for medical image enhancement
- Multi-channel tensor conversion for improved segmentation
- U-Net model training with custom accuracy metrics
- Model export for inference

Dependencies:
- fastai: Deep learning framework
- nibabel: NIfTI file handling
- numpy, pandas: Data manipulation
- matplotlib: Visualization
- torch: PyTorch backend

Author: Liver Tumor Segmentation Team
Date: 2024
License: MIT
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

import nibabel as nib
import cv2
import imageio
from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image

import fastai
from fastai.vision.all import *
from fastai.data.transforms import *

import gc

# Create a meta file for nii files processing
file_list = []
for directory, _, filenames in os.walk('../input/liver-tumor-segmentation'):
    for filename in filenames:
        file_list.append((directory, filename))

for directory, _, filenames in os.walk('../input/liver-tumor-segmentation-part-2'):
    for filename in filenames:
        file_list.append((directory, filename))

metadata_df = pd.DataFrame(file_list, columns=['directory', 'filename'])
metadata_df.sort_values(by=['filename'], ascending=True)

# Map CT scan and label
metadata_df["mask_directory"] = ""
metadata_df["mask_filename"] = ""

for i in range(131):
    ct_file = f"volume-{i}.nii"
    mask_file = f"segmentation-{i}.nii"

    metadata_df.loc[metadata_df['filename'] == ct_file, 'mask_filename'] = mask_file
    metadata_df.loc[metadata_df['filename'] == ct_file, 'mask_directory'] = "../input/liver-tumor-segmentation/segmentations"

test_data = metadata_df[metadata_df.mask_filename == '']
# Drop rows without masks
metadata_df = metadata_df[metadata_df.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True)
print(len(metadata_df))
metadata_df

def read_nii(file_path):
    """
    Read NIfTI (.nii) file and return pixel array.
    
    This function loads a NIfTI file using nibabel and converts it to a numpy array
    with appropriate rotation for medical imaging visualization.
    
    Args:
        file_path (str): Path to the NIfTI file to be loaded
        
    Returns:
        numpy.ndarray: 3D pixel array representing the CT scan data with shape (height, width, slices)
        
    Note:
        - The array is rotated 90 degrees counterclockwise for proper orientation
        - Compatible with medical imaging standards (DICOM orientation)
    """
    ct_scan = nib.load(file_path)
    pixel_array = ct_scan.get_fdata()
    pixel_array = np.rot90(np.array(pixel_array))
    return pixel_array

# Read a sample .nii file
sample_index = 3
sample_ct_scan = read_nii(metadata_df.loc[sample_index, 'directory'] + "/" + metadata_df.loc[sample_index, 'filename'])
sample_mask = read_nii(metadata_df.loc[sample_index, 'mask_directory'] + "/" + metadata_df.loc[sample_index, 'mask_filename'])
print(sample_ct_scan.shape)
print(sample_mask.shape)
print(metadata_df.loc[sample_index, 'directory'] + "/" + metadata_df.loc[sample_index, 'filename'])
print(np.amin(sample_ct_scan), np.amax(sample_ct_scan))
print(np.amin(sample_mask), np.amax(sample_mask))

# Define DICOM window settings for different anatomical regions
# These values are medically optimized for visualizing specific tissues and organs
dicom_windows = types.SimpleNamespace(
    brain=(80, 40),           # Brain tissue visualization
    subdural=(254, 100),      # Subdural space
    stroke=(8, 32),           # Stroke detection
    brain_bone=(2800, 600),   # Brain with bone structures
    brain_soft=(375, 40),     # Soft brain tissue
    lungs=(1500, -600),       # Lung tissue
    mediastinum=(350, 50),    # Mediastinal structures
    abdomen_soft=(400, 50),   # Abdominal soft tissue
    liver=(150, 30),          # Liver tissue (primary focus)
    spine_soft=(250, 50),     # Spinal soft tissue
    spine_bone=(1800, 400),   # Spinal bone structures
    custom=(200, 60)          # Custom window for liver tumor segmentation
)

@patch
def apply_window(self: Tensor, window_width, window_level):
    """
    Apply DICOM windowing to a tensor for medical image enhancement.
    
    This function applies windowing (contrast adjustment) to medical images
    to enhance visualization of specific tissues. Windowing maps the CT
    Hounsfield units to a displayable grayscale range.
    
    Args:
        self (Tensor): Input tensor containing CT scan data
        window_width (int): Range of Hounsfield units to display
        window_level (int): Center value of the window
        
    Returns:
        Tensor: Windowed tensor with values normalized to [0, 1]
        
    Note:
        - Window width determines contrast (wider = lower contrast)
        - Window level determines brightness (center of intensity range)
        - Values outside the window are clipped to min/max
    """
    pixel_data = self.clone()
    min_value = window_level - window_width // 2
    max_value = window_level + window_width // 2
    pixel_data[pixel_data < min_value] = min_value
    pixel_data[pixel_data > max_value] = max_value
    return (pixel_data - min_value) / (max_value - min_value)

# Plot a sample image
plt.imshow(tensor(sample_ct_scan[..., 50].astype(np.float32)).apply_window(*dicom_windows.liver), cmap=plt.cm.bone)

# Function to plot a sample image
def plot_sample(array_list, color_map='nipy_spectral'):
    """
    Plot multiple medical image arrays in a grid layout for comparison.
    
    This function displays medical images including original CT scans, windowed versions,
    and segmentation masks in a 4-panel subplot for visualization and analysis.
    
    Args:
        array_list (list): List containing medical image arrays:
                          [0] - Original CT scan slice
                          [1] - Corresponding segmentation mask
        color_map (str, optional): Matplotlib colormap for mask visualization. 
                                 Defaults to 'nipy_spectral'
                                 
    Returns:
        None: Displays the plot using matplotlib
        
    Note:
        - Panel 1: Original CT scan image
        - Panel 2: Windowed CT scan (liver window applied)
        - Panel 3: Segmentation mask overlay
        - Panel 4: Mask visualization with specified colormap
    """
    fig = plt.figure(figsize=(18, 15))

    plt.subplot(1, 4, 1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(tensor(array_list[0].astype(np.float32)).apply_window(*dicom_windows.liver), cmap='bone')
    plt.title('Windowed Image')

    plt.subplot(1, 4, 3)
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Mask')

    plt.subplot(1, 4, 4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Liver & Mask')

    plt.show()

sample_slice = 410  # this is basically the slice
sample_slice_data = tensor(sample_ct_scan[..., sample_slice].astype(np.float32))

plot_sample([sample_ct_scan[..., sample_slice], sample_mask[..., sample_slice]])

# Check the mask values
mask_image = Image.fromarray(sample_mask[..., sample_slice].astype('uint8'), mode="L")
print(mask_image.size)
unique, counts = np.unique(mask_image, return_counts=True)
print(np.array((unique, counts)).T)
plt.imshow(mask_image, cmap='bone')

# Preprocessing functions
# Source: https://docs.fast.ai/medical.imaging

class CTScanTensor(TensorImageBW):
    """
    Custom tensor class for CT scan medical imaging data.
    
    This class extends TensorImageBW to provide specialized functionality for
    CT scan data processing and visualization. It includes default display
    settings optimized for medical imaging.
    
    Attributes:
        _show_args (dict): Default visualization arguments using bone colormap
                          for medical image display
                          
    Note:
        - Inherits from TensorImageBW for grayscale medical image handling
        - Bone colormap is standard for CT scan visualization
        - Optimized for FastAI medical imaging workflow
    """
    _show_args = {'cmap': 'bone'}

# Function to calculate histogram bins
@patch
def compute_histogram_bins(self: Tensor, num_bins=100):
    """
    Compute adaptive histogram bins for tensor data normalization.
    
    This function calculates histogram bins based on the distribution of pixel
    intensities in the tensor, enabling better contrast enhancement for medical
    images with varying intensity ranges.
    
    Args:
        self (Tensor): Input tensor containing medical image data
        num_bins (int, optional): Number of histogram bins to compute. Defaults to 100
        
    Returns:
        Tensor: Computed histogram bin boundaries for intensity mapping
        
    Note:
        - Uses quantile-based binning for robust intensity distribution
        - Handles outliers by clipping to 0.1% and 99.9% percentiles
        - Essential for histogram equalization in medical imaging
    """
    sorted_data = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(num_bins).float() / num_bins + (1 / 2 / num_bins),
                   tensor([0.999])])
    t = (len(sorted_data) * t).long()
    return sorted_data[t].unique()

# Function to scale a tensor using histogram bins
@patch
def scale_using_histogram(self: Tensor, bins=None):
    """
    Scale tensor values using histogram equalization for improved contrast.
    
    This function performs histogram equalization by mapping pixel intensities
    to a uniform distribution, improving contrast in medical images with
    poor visibility or non-uniform illumination.
    
    Args:
        self (Tensor): Input tensor containing medical image data
        bins (Tensor, optional): Pre-computed histogram bins. If None, 
                                computes bins automatically
        
    Returns:
        Tensor: Histogram-equalized tensor with values in range [0, 1]
        
    Note:
        - Automatically detects CUDA tensors for GPU acceleration
        - Uses linear interpolation for smooth intensity mapping
        - Clamps output values to ensure valid intensity range
    """
    if self.device.type == 'cuda':
        return self.scale_using_histogram_cuda(bins)
    if bins is None:
        bins = self.compute_histogram_bins()
    ys = np.linspace(0., 1., len(bins))
    x = self.numpy().flatten()
    x = np.interp(x, bins.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)

# Function to convert a tensor to multiple channels
@patch
def convert_to_multichannel(self: Tensor, windows, bins=None):
    """
    Convert single-channel CT scan to multi-channel tensor for enhanced segmentation.
    
    This function creates a multi-channel representation by applying different
    DICOM windows and histogram equalization, providing the model with multiple
    perspectives of the same anatomical structures.
    
    Args:
        self (Tensor): Input single-channel CT scan tensor
        windows (list): List of (window_width, window_level) tuples for different 
                       tissue visualizations
        bins (Tensor or int, optional): Histogram bins for equalization. 
                                      If 0, skips histogram channel
        
    Returns:
        CTScanTensor: Multi-channel tensor with windowed and histogram-equalized views
        
    Note:
        - Each channel represents different tissue contrast
        - Histogram channel provides global contrast information
        - Essential for improving segmentation accuracy
    """
    result = [self.apply_window(*window) for window in windows]
    if not isinstance(bins, int) or bins != 0:
        result.append(self.scale_using_histogram(bins).clamp(0, 1))
    dim = [0, 1][self.dim() == 3]
    return CTScanTensor(torch.stack(result, dim=dim))

# Function to save an image as a JPG
@patch
def save_as_jpg(self: Tensor, file_path, windows, bins=None, quality=90):
    """
    Save tensor as multi-channel JPEG image for training data preparation.
    
    This function converts a CT scan tensor to a multi-channel JPEG image
    suitable for training deep learning models. It applies windowing and
    histogram equalization before saving.
    
    Args:
        self (Tensor): Input CT scan tensor
        file_path (str or Path): Output file path (extension will be changed to .jpg)
        windows (list): List of DICOM window settings for multi-channel conversion
        bins (Tensor or int, optional): Histogram bins for equalization. Defaults to None
        quality (int, optional): JPEG compression quality (0-100). Defaults to 90
        
    Returns:
        None: Saves the image to disk
        
    Note:
        - Automatically converts tensor values to 0-255 range
        - Supports RGB (3-channel) and CMYK (4-channel) modes
        - High quality setting preserves medical image details
    """
    file_name = Path(file_path).with_suffix('.jpg')
    self = (self.convert_to_multichannel(windows, bins) * 255).byte()
    image = Image.fromarray(self.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][self.shape[0] == 4])
    image.save(file_name, quality=quality)

# Test image save function
_, axes = subplots(1, 1)
sample_slice_data.save_as_jpg('test.jpg', [dicom_windows.liver, dicom_windows.custom])
show_image(Image.open('test.jpg'), ax=axes[0])
show_image(Image.open('test.png'), ax=axes[0])

# Select a subset of files for training
training_files = metadata_df[100:111]
training_files

# Generate JPG files for Unet training
GENERATE_JPG_FILES = True  # Warning: generation takes ~ 1 hour
total_slices = 0
if GENERATE_JPG_FILES:
    os.makedirs('train_images', exist_ok=True)
    os.makedirs('train_masks', exist_ok=True)

    for i in tqdm(range(100 + 0, 100 + len(training_files))):  # Take 1/3 nii files for training
        current_ct_scan = read_nii(training_files.loc[i, 'directory'] + "/" + training_files.loc[i, 'filename'])
        current_mask = read_nii(training_files.loc[i, 'mask_directory'] + "/" + training_files.loc[i, 'mask_filename'])
        current_file_name = str(training_files.loc[i, 'filename']).split('.')[0]
        current_dimension = current_ct_scan.shape[2]  # 512, 512, current_dimension
        total_slices = total_slices + current_dimension

        for current_slice in range(0, current_dimension, 1):  # Export every 2nd slice for training
            image_data = tensor(current_ct_scan[..., current_slice].astype(np.float32))
            mask_image = Image.fromarray(current_mask[..., current_slice].astype('uint8'), mode="L")
            image_data.save_as_jpg(f"train_images/{current_file_name}_slice_{current_slice}.jpg", [dicom_windows.liver, dicom_windows.custom])
            mask_image.save(f"train_masks/{current_file_name}_slice_{current_slice}_mask.png")
else:
    path = Path("../input/liver-segmentation-with-fastai-v2")  # Read JPG from saved kernel output
print(total_slices)

# MODEL TRAINING
batch_size = 16
image_size = 128

class_names = np.array(["background", "liver", "tumor"])

# Function to get the input image
def get_input_image(file_path: Path):
    """
    Get the input image file path for FastAI data loading.
    
    This is a simple identity function used by FastAI's DataBlock to specify
    how to obtain the input image from a file path.
    
    Args:
        file_path (Path): Path to the input image file
        
    Returns:
        Path: The same file path (identity function)
        
    Note:
        - Used in FastAI DataBlock configuration
        - Enables flexible data loading pipeline
    """
    return file_path

# Function to get the label for the image
def get_label(file_path: Path):
    """
    Get the corresponding segmentation mask path for an input image.
    
    This function constructs the path to the segmentation mask that corresponds
    to a given input image, following the naming convention used in the dataset.
    
    Args:
        file_path (Path): Path to the input image file
        
    Returns:
        Path: Path to the corresponding segmentation mask file
        
    Note:
        - Assumes mask files are in 'train_masks' directory
        - Mask files have '_mask.png' suffix
        - Essential for supervised learning setup
    """
    return path / 'train_masks' / f'{file_path.stem}_mask.png'

transformations = [IntToFloatTensor(), Normalize()]

data_block = DataBlock(blocks=(ImageBlock(), MaskBlock(class_names)),
               batch_tfms=transformations,
               splitter=RandomSplitter(),
               item_tfms=[Resize(image_size)],
               get_items=get_image_files,
               get_y=get_label
              )

datasets = data_block.datasets(source='./train_images')
print(len(datasets))
print(datasets)

index = 20
images = [datasets[index][0], datasets[index][1]]
fig, axes = plt.subplots(1, 2)
for i, ax in enumerate(axes.flatten()):
    ax.axis('off')
    ax.imshow(images[i])

unique_labels, counts = np.unique(array(datasets[index][1]), return_counts=True)
print(np.array((unique_labels, counts)).T)

data_loaders = data_block.dataloaders(path/'train_images', bs=batch_size)
data_loaders.show_batch()

# Custom accuracy functions
def non_background_accuracy(prediction, target, background_index=0, axis=1):
    """
    Calculate accuracy excluding background pixels for medical segmentation.
    
    This metric focuses on the accuracy of tissue and tumor segmentation by
    excluding background pixels, which are typically abundant and can skew
    overall accuracy metrics in medical imaging.
    
    Args:
        prediction (Tensor): Model predictions with shape (batch, classes, height, width)
        target (Tensor): Ground truth segmentation masks
        background_index (int, optional): Index of background class. Defaults to 0
        axis (int, optional): Axis along which to compute argmax. Defaults to 1
        
    Returns:
        Tensor: Mean accuracy for non-background pixels
        
    Note:
        - Essential for medical segmentation where background dominates
        - Provides more meaningful accuracy for clinical applications
        - Excludes class imbalance bias from background pixels
    """
    target = target.squeeze(1)
    mask = target != background_index
    return (prediction.argmax(dim=axis)[mask] == target[mask]).float().mean() 

def custom_non_background_accuracy(prediction, target):
    """
    Custom wrapper for non-background accuracy with specific parameters.
    
    This function provides a specialized accuracy metric for liver tumor
    segmentation with adjusted background handling.
    
    Args:
        prediction (Tensor): Model predictions
        target (Tensor): Ground truth segmentation masks
        
    Returns:
        Tensor: Custom non-background accuracy value
        
    Note:
        - Uses background_index=3 as a dummy value for specific requirements
        - Wrapper function for integration with FastAI metrics
    """
    return non_background_accuracy(prediction=prediction, target=target, background_index=3, axis=1)  # 3 is a dummy value to include the background which is 0

# Create a UNet learner
learner = unet_learner(data_loaders, resnet34, loss_func=CrossEntropyLossFlat(axis=1), metrics=[non_background_accuracy, custom_non_background_accuracy])

# Find the learning rate
learner.lr_find()

# Fine-tune the model
learner.fine_tune(5, wd=0.1, cbs=SaveModelCallback())

# Show model results
learner.show_results()

# Save the trained model
learner.export(path / 'Liver_segmentation')

gc.collect()
torch.cuda.empty_cache()
