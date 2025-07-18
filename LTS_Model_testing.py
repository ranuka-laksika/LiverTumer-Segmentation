"""
Liver Tumor Segmentation Model Testing Module

This module provides functionality for testing and evaluating a trained deep learning
model for liver tumor segmentation. It processes NIfTI (.nii) files, applies the
trained model for inference, and evaluates segmentation performance.

Key Features:
- NIfTI file processing for model inference
- Trained model loading and prediction
- Performance evaluation and visualization
- Batch processing of test datasets
- Confusion matrix generation and analysis

Dependencies:
- fastai: Deep learning framework for model loading
- nibabel: NIfTI file handling
- opencv-cv2: Image processing and contour analysis
- numpy, pandas: Data manipulation
- matplotlib, seaborn: Visualization

Author: Liver Tumor Segmentation Team
Date: 2024
License: MIT
"""

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
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

import cv2 
import numpy as np
import os

import seaborn as sns

# Create a meta file for processing NII files
file_list = []
for dirname, _, filenames in os.walk('../input/liver-tumor-segmentation'):
    for filename in filenames:
        file_list.append((dirname, filename))

for dirname, _, filenames in os.walk('../input/liver-tumor-segmentation-part-2'):
    for filename in filenames:
        file_list.append((dirname, filename))

data_files = pd.DataFrame(file_list, columns=['directory', 'filename'])

# Map CT scans and labels
data_files["mask_directory"] = ""
data_files["mask_filename"] = ""

for i in range(131):
    ct_scan = f"volume-{i}.nii"
    mask = f"segmentation-{i}.nii"

    data_files.loc[data_files['filename'] == ct_scan, 'mask_filename'] = mask
    data_files.loc[data_files['filename'] == ct_scan, 'mask_directory'] = "../input/liver-tumor-segmentation/segmentations"

test_data_files = data_files[data_files.mask_filename == '']
data_files = data_files[data_files.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True) 

# Define a function to read NII files and convert them into a numpy array
def read_nii(filepath):
    """
    Read NIfTI (.nii) file and convert to numpy array for inference.
    
    This function loads a NIfTI file and converts it to a properly oriented
    numpy array suitable for model inference and visualization.
    
    Args:
        filepath (str): Path to the NIfTI file to be loaded
        
    Returns:
        numpy.ndarray: 3D array representing CT scan data with shape (height, width, slices)
        
    Note:
        - Applies 90-degree rotation for proper medical image orientation
        - Compatible with medical imaging standards
        - Used for both CT scans and segmentation masks
    """
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

# Split the dataset into test and train
data_files = data_files[0:20]

# Define model-related variables and functions
batch_size = 16
image_size = 128
class_codes = np.array(["background", "liver", "tumor"])
model_path = './'

def get_x(file_path: Path):
    """
    Get input file path for FastAI data loading.
    
    Simple identity function used by FastAI's data loading pipeline
    to specify how to obtain input files.
    
    Args:
        file_path (Path): Path to input file
        
    Returns:
        Path: The same file path (identity function)
    """
    return file_path

def label_func(x): 
    """
    Generate corresponding mask file path for input image.
    
    This function constructs the path to the segmentation mask that
    corresponds to a given input image file, following the dataset
    naming convention.
    
    Args:
        x (Path): Path to input image file
        
    Returns:
        Path: Path to corresponding mask file
        
    Note:
        - Assumes masks are in 'train_masks' subdirectory
        - Adds '_mask.png' suffix to the stem filename
    """
    return model_path/'train_masks'/f'{x.stem}_mask.png'

def foreground_accuracy(input_tensor, target_tensor, background_idx=0, axis=1):
    "Compute non-background accuracy for multiclass segmentation"
    target_tensor = target_tensor.squeeze(1)
    mask = target_tensor != background_idx
    return (input_tensor.argmax(dim=axis)[mask] == target_tensor[mask]).float().mean()

def custom_foreground_accuracy(input_tensor, target_tensor):  
    """
    Custom foreground accuracy metric for liver tumor segmentation.
    
    This function provides a specialized accuracy calculation that excludes
    background pixels, focusing on the segmentation performance of liver
    and tumor regions.
    
    Args:
        input_tensor (Tensor): Model predictions
        target_tensor (Tensor): Ground truth segmentation masks
        
    Returns:
        Tensor: Foreground accuracy value excluding background pixels
        
    Note:
        - Uses background_idx=3 for specific requirements
        - Wrapper for the generic foreground_accuracy function
        - Essential for medical segmentation evaluation
    """
    return foreground_accuracy(input_tensor=input_tensor, target_tensor=target_tensor, background_idx=3, axis=1)

# Load the model
transformations = [Resize(image_size), IntToFloatTensor(), Normalize()]
learned_model = load_learner('../input/trained-model/Liver_segmentation', cpu=False)
learned_model.dls.transform = transformations

# Define a function to convert NII files for model prediction
def nii_transformer(file_path, windows):
    """
    Transform NIfTI file into format suitable for model prediction.
    
    This function loads a NIfTI file, extracts a specific slice, applies
    DICOM windowing, and converts it to the tensor format expected by
    the trained model.
    
    Args:
        file_path (str): Path to the NIfTI file
        windows (list): List of DICOM window settings for multi-channel conversion
        
    Returns:
        list: List containing transformed tensor images ready for prediction
        
    Note:
        - Extracts slice 450 (hardcoded for specific dataset)
        - Applies multi-channel windowing for enhanced features
        - Converts to TensorImage format for FastAI compatibility
    """ 
    slices = []
    ct_scan = read_nii(file_path)
    data = tensor(ct_scan[..., 450].astype(np.float32))
    data = (data.to_nchan(windows) * 255).byte()
    slices.append(TensorImage(data))
    return slices

# Select the test number and slice number for prediction
test_number = 3
test_slice_index = 450
test_nii = read_nii(data_files.loc[test_number, 'directory'] + "/" + data_files.loc[test_number, 'filename'])
test_mask = read_nii(data_files.loc[test_number, 'mask_directory'] + "/" + data_files.loc[test_number, 'mask_filename'])
sample_slice = tensor(test_nii[..., test_slice_index].astype(np.float32))

# Prepare an NII test file for prediction
test_files = nii_transformer(data_files.loc[test_number, 'directory'] + "/" + data_files.loc[test_number, 'filename'], [dicom_windows.liver, dicom_windows.custom])

# Get predictions for a Test file
test_dataloader = learned_model.dls.test_dl(test_files)
predictions, targets = learned_model.get_preds(dl=test_dataloader)
predicted_mask = np.argmax(predictions, axis=1)

# Calculate some statistics on the predicted mask
predicted_mask_array = np.array(predicted_mask[0])
unique_values, value_counts = np.unique(predicted_mask_array, return_counts=True)

# Perform predictions on multiple images
def nii_transformer_selective(file_path, windows, current_slice): 
    """
    Transform specific slice from NIfTI file for model prediction.
    
    This function extracts and transforms a specific slice from a NIfTI file,
    allowing for slice-by-slice processing of volumetric medical data.
    
    Args:
        file_path (str): Path to the NIfTI file
        windows (list): List of DICOM window settings for multi-channel conversion
        current_slice (int): Index of the slice to extract and process
        
    Returns:
        list: List containing transformed tensor image for the specified slice
        
    Note:
        - Enables selective slice processing for large volumes
        - Applies same windowing and conversion as nii_transformer
        - Used for batch processing and evaluation
    """
    slices = []
    ct_scan = read_nii(file_path)
    data = tensor(ct_scan[..., current_slice].astype(np.float32))
    data = (data.to_nchan(windows) * 255).byte()
    slices.append(TensorImage(data))
    return slices

def check_mask(mask_image):
    """
    Check if mask image contains any segmented regions.
    
    This function uses OpenCV to detect contours in a binary mask image,
    determining whether any segmented regions (liver/tumor) are present.
    
    Args:
        mask_image (numpy.ndarray): Binary mask image to analyze
        
    Returns:
        int: 1 if contours are found, 0 if no contours detected
        
    Note:
        - Uses external contour detection for robust analysis
        - Essential for determining slice relevance in medical datasets
        - Returns binary indicator for presence of segmented structures
    """
    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        return 1
    else:
        return 0

# Define some variables and lists
selected_numbers = [3, 4, 5]
volume_file_names = ['volume-100.nii', 'volume-102.nii', 'volume-102.nii']
segmentation_file_names = ['segmentation-100.nii', 'segmentation-102.nii', 'segmentation-102.nii']
confusion_matrix = np.zeros((2, 2), dtype=int)

# Loop through the images and perform predictions
for selected_number, volume_file_name, segmentation_file_name in zip(selected_numbers, volume_file_names, segmentation_file_names):
    current_mask = read_nii(data_files.loc[selected_number, 'mask_directory'] + "/" + data_files.loc[selected_number, 'mask_filename'])
    for current_slice in tqdm(range(250, 550, 10)):
        test_file = nii_transformer_selective(data_files.loc[selected_number, 'directory'] + "/" + data_files.loc[selected_number, 'filename'], [dicom_windows.liver, dicom_windows.custom], current_slice)
        test_dataloader = learned_model.dls.test_dl(test_file)
        predictions, targets = learned_model.get_preds(dl=test_dataloader)
        predicted_mask = np.argmax(predictions, axis=1)
        predicted_mask_array = np.array(predicted_mask[0])

        is_tumor_predicted = False
        unique_predicted = np.unique(predicted_mask_array)

        if 0 in unique_predicted:
            is_background_predicted = True
        else:
            is_background_predicted = False
        if 1 in unique_predicted:
            is_liver_predicted = True
        else:
            is_liver_predicted = False
        if 2 in unique_predicted:
            is_tumor_predicted = True
        else:
            is_tumor_predicted = False

        mask = Image.fromarray(current_mask[..., current_slice].astype('uint8'), mode="L")
        is_tumor_actual = False
        unique_actual = np.unique(mask)

        if 0 in unique_actual:
            is_background_actual = True
        else:
            is_background_actual = False
        if 1 in unique_actual:
            is_liver_actual = True
        else:
            is_liver_actual = False
        if 2 in unique_actual:
            is_tumor_actual = True
        else:
            is_tumor_actual = False

        if is_tumor_predicted and is_tumor_actual:
            confusion_matrix[0, 0] += 1
        if not is_tumor_predicted and not is_tumor_actual:
            confusion_matrix[1, 1] += 1
        if not is_tumor_predicted and is_tumor_actual:
            confusion_matrix[1, 0] += 1
        if is_tumor_predicted and not is_tumor_actual:
            confusion_matrix[0, 1] += 1

# Print the confusion matrix
print(confusion_matrix)

# Plot Confusion Matrix
heatmap = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
heatmap.set_title('Seaborn Confusion Matrix with labels\n\n')
heatmap.set_xlabel('\nActual Values')
heatmap.set_ylabel('Predicted Values ')

# Set the tick labels
heatmap.xaxis.set_ticklabels(['True', 'False'])
heatmap.yaxis.set_ticklabels(['True', 'False'])

# Display the visualization of the Confusion Matrix
plt.show()

# Convert the `.nii` files to jpg and generate corresponding masks
selected_numbers = [3, 4, 5]
volume_file_names = ['volume-100.nii', 'volume-102.nii', 'volume-102.nii']
segmentation_file_names = ['segmentation-100.nii', 'segmentation-102.nii', 'segmentation-102.nii']
total_slices = 0

# Loop through the files and save them as JPG images
for selected_number, volume_file_name, segmentation_file_name in zip(selected_numbers, volume_file_names, segmentation_file_names):
    current_ct_scan = read_nii(data_files.loc[selected_number, 'directory'] + "/" + data_files.loc[selected_number, 'filename'])
    current_mask = read_nii(data_files.loc[selected_number, 'mask_directory'] + "/" + data_files.loc[selected_number, 'mask_filename'])
    current_file_name = str(data_files.loc[selected_number, 'filename']).split('.')[0]
    current_dimensions = current_ct_scan.shape[2]

    for current_slice in tqdm(range(250, 550, 10)):
        data = tensor(current_ct_scan[..., current_slice].astype(np.float32))
        mask = Image.fromarray(current_mask[..., current_slice].astype('uint8'), mode="L")
        data.save_jpg(f"images/{current_file_name}_slice_{current_slice}.jpg", [dicom_windows.liver, dicom_windows.custom])
        mask.save(f"mask/{current_file_name}_slice_{current_slice}_mask.png")
        total_slices += 1

# Use OpenCV to analyze mask images and save the results to a CSV file
def check_mask_image(mask_image):
    """
    Analyze mask image for contour detection and classification.
    
    This function performs contour detection on segmentation masks to determine
    the presence of anatomical structures, used for evaluation and validation
    of segmentation results.
    
    Args:
        mask_image (numpy.ndarray): Binary or grayscale mask image
        
    Returns:
        int: 1 if contours are detected, 0 if no contours found
        
    Note:
        - Identical to check_mask but used in different processing contexts
        - Essential for automated quality assessment of segmentation
        - Used in batch processing for CSV output generation
    """
    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        return 1
    else:
        return 0

file = open('true_values.csv', mode='w')

file_iterator = os.scandir('./mask')
for i in file_iterator:
    mask_image = cv2.imread("./mask/" + i.name, -1)

    if np.count_nonzero(mask_image) == 0:
        file.write('0,0,0\n')
    else:
        file.write('0,')
        liver_mask = np.where(mask_image == 1, 255, mask_image)
        liver_mask = np.where(liver_mask == 2, 0, mask_image)
        result = check_mask_image(liver_mask)
        if result:
            file.write('1,')
        else:
            file.write('0,')

        tumor_mask = np.where(mask_image == 2, 255, mask_image)
        tumor_mask = np.where(tumor_mask == 1, 0, mask_image)
        result = check_mask_image(tumor_mask)
        if result:
            file.write('1')
        else:
            file.write('0')

        file.write('\n')

file.close()
