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

# List all installed packages
!pip list

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

# Read a function to read .nii files
def read_nii(file_path):
    '''
    Reads .nii file and returns pixel array
    '''
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

# Define DICOM window settings
dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(150, 30),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60)
)

@patch
def apply_window(self: Tensor, window_width, window_level):
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

plot_sample([sample_ct_scan[..., sample_slice], sample_mask[..., sample_slice])

# Check the mask values
mask_image = Image.fromarray(sample_mask[..., sample_slice].astype('uint8'), mode="L")
print(mask_image.size)
unique, counts = np.unique(mask_image, return_counts=True)
print(np.array((unique, counts)).T)
plt.imshow(mask_image, cmap='bone')

# Preprocessing functions
# Source: https://docs.fast.ai/medical.imaging

# Define a class to represent a CT scan tensor
class CTScanTensor(TensorImageBW):
    _show_args = {'cmap': 'bone'}

# Function to calculate histogram bins
@patch
def compute_histogram_bins(self: Tensor, num_bins=100):
    sorted_data = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(num_bins).float() / num_bins + (1 / 2 / num_bins),
                   tensor([0.999])])
    t = (len(sorted_data) * t).long()
    return sorted_data[t].unique()

# Function to scale a tensor using histogram bins
@patch
def scale_using_histogram(self: Tensor, bins=None):
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
    result = [self.apply_window(*window) for window in windows]
    if not isinstance(bins, int) or bins != 0:
        result.append(self.scale_using_histogram(bins).clamp(0, 1))
    dim = [0, 1][self.dim() == 3]
    return CTScanTensor(torch.stack(result, dim=dim)

# Function to save an image as a JPG
@patch
def save_as_jpg(self: Tensor, file_path, windows, bins=None, quality=90):
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

    for i in tqdm(range(100 + 0, 100 + len(training_files)):  # Take 1/3 nii files for training
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
    return file_path

# Function to get the label for the image
def get_label(file_path: Path):
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
    target = target.squeeze(1)
    mask = target != background_index
    return (prediction.argmax(dim=axis)[mask] == target[mask]).float().mean() 

def custom_non_background_accuracy(prediction, target):
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
