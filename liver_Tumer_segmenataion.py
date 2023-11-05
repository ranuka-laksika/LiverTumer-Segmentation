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


base_path = r'C:\Users\Ramudi\Desktop\Project_01' 
# Define Constants
DICOM_WINDOWS = {
    "liver": (150, 30),
    "custom": (200, 60),
    
}

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

def preprocess_image(image, window, level):
  
    min_value = level - (window / 2)
    max_value = level + (window / 2)

    # Clip and normalize the image to the specified window and level
    preprocessed_image = np.clip(image, min_value, max_value)
    preprocessed_image = (preprocessed_image - min_value) / (max_value - min_value)

    return preprocessed_image


def generate_jpg_files(df, output_dir, generate_masks=False):
    os.makedirs(output_dir, exist_ok=True)
    slice_sum = 0

    for index, row in df.iterrows():
        curr_ct = read_nii(os.path.join(row['dirname'], row['filename']))
        curr_dim = curr_ct.shape[2]
        slice_sum += curr_dim

        for curr_slice in range(0, curr_dim, 1):
            data = curr_ct[..., curr_slice].astype(np.float32)
            data = preprocess_image(data, DICOM_WINDOWS['liver'], DICOM_WINDOWS['custom'])

            data_filename = f"{row['filename'].split('.')[0]}_slice_{curr_slice}.jpg"
            data.save_jpg(os.path.join(output_dir, data_filename))

            if generate_masks:
                curr_mask = read_nii(os.path.join(row['mask_dirname'], row['mask_filename']))
                mask = Image.fromarray(curr_mask[..., curr_slice].astype('uint8'), mode="L")
                mask_filename = f"{row['filename'].split('.')[0]}_slice_{curr_slice}_mask.png"
                mask.save(os.path.join(output_dir, mask_filename))

    return slice_sum

# Example usage
df_files = pd.DataFrame(...)  # Define your DataFrame
output_dir = "train_images"
generate_masks = True  # Set to True if you want to generate mask images

slice_sum = generate_jpg_files(df_files, output_dir, generate_masks)
print(f"Total slices: {slice_sum}")

# MODEL TRAINING
bs = 16
im_size = 128

codes = np.array(["background", "liver", "tumor"])

def get_x(fname: Path):
    return fname

def label_func(x):
    return path / 'train_masks' / f'{x.stem}_mask.png'

tfms = [IntToFloatTensor(), Normalize()]

db = DataBlock(
    blocks=(ImageBlock(), MaskBlock(codes)),
    batch_tfms=tfms,
    splitter=RandomSplitter(),
    item_tfms=[Resize(im_size)],
    get_items=get_image_files,
    get_y=label_func
)

ds = db.datasets(source='./train_images')
print(len(ds))
print(ds)

idx = 20
imgs = [ds[idx][0], ds[idx][1]
fig, axs = plt.subplots(1, 2)
for i, ax in enumerate(axs.flatten()):
    ax.axis('off')
    ax.imshow(imgs[i])

unique, counts = np.unique(array(ds[idx][1]), return_counts=True)
print(np.array((unique, counts)).T)

dls = db.dataloaders(path / 'train_images', bs=bs)

def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()

def cust_foreground_acc(inp, targ):
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)

learn = unet_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(axis=1), metrics=[foreground_acc, cust_foreground_acc])

learn.lr_find()
learn.fine_tune(5, wd=0.1, cbs=SaveModelCallback())
learn.show_results()

# Save the model
learn.export(path / 'Liver_segmentation')

import gc
del learn
gc.collect()
torch.cuda.empty_cache()

# TESTING MODEL

print(fastai.__version__)

# Define the path
path = './'

# Load saved model
bs = 16
im_size = 128

# Define labels for classes
codes = np.array(["background", "liver", "tumor"])

# Define transform functions
def get_x(fname: Path):
    return fname

def label_func(x):
    return path / 'train_masks' / f'{x.stem}_mask.png'

def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()

def cust_foreground_acc(inp, targ):
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)

# Load the model
tfms = [Resize(im_size), IntToFloatTensor(), Normalize()]
learn0 = load_learner('../input/trained-model/Liver_segmentation', cpu=False)
learn0.dls.transform = tfms

# Function to convert nii file to the format used by the model
def nii_tfm(fn, wins):
    test_nii = read_nii(fn)
    curr_dim = test_nii.shape[2]
    slices = []

    data = tensor(test_nii[..., 450].astype(np.float32))
    data = (data.to_nchan(wins) * 255).byte()
    slices.append(TensorImage(data))

    return slices

# Select test number and slice number
tst = 3
test_slice_idx = 450
test_nii = read_nii(df_files.loc[tst, 'dirname'] + "/" + df_files.loc[tst, 'filename'])
test_mask = read_nii(df_files.loc[tst, 'mask_dirname'] + "/" + df_files.loc[tst, 'mask_filename'])
sample_slice = tensor(test_nii[..., test_slice_idx].astype(np.float32))

# Prepare a nii test file for prediction
test_files = nii_tfm(df_files.loc[tst, 'dirname'] + "/" + df_files.loc[tst, 'filename'], [dicom_windows.liver, dicom_windows.custom])

# Get predictions for the test file
test_dl = learn0.dls.test_dl(test_files)
preds, y = learn0.get_preds(dl=test_dl)

predicted_mask = np.argmax(preds, axis=1)
plt.imshow(predicted_mask[0])
a = np.array(predicted_mask[0])

unique, counts = np.unique(a, return_counts=True)
print(np.array((unique, counts)).T)

# Define a function to convert nii files to jpg and masks
def nii_tfm_selctive(fn, wins, curr_slice):
    slices = []
    test_nii = read_nii(fn)
    data = tensor(test_nii[..., curr_slice].astype(np.float32))
    data = (data.to_nchan(wins) * 255).byte()
    slices.append(TensorImage(data))
    return slices

# Define some variables for testing
nums = [3, 4, 5]
vol_names = ['volume-100.nii', 'volume-102.nii', 'volume-102.nii']
seg_names = ['segmentation-100.nii', 'segmentation-102.nii', 'segmentation-102.nii']

# Initialize a confusion matrix
conf_matrix = np.zeros((2, 2), dtype=int)

# Loop through the data for testing
for nums, vol_names, seg_names in zip(nums, vol_names, seg_names):
    curr_mask = read_nii(df_files.loc[nums, 'mask_dirname'] + "/" + df_files.loc[nums, 'mask_filename'])
    for curr_slice in tqdm(range(250, 550, 10)):  # Export every 10th slice for testing
        # For prediction
        test_file = nii_tfm_selctive(df_files.loc[nums, 'dirname'] + "/" + df_files.loc[nums, 'filename'],
                                     [dicom_windows.liver, dicom_windows.custom], curr_slice)
        test_dl = learn0.dls.test_dl(test_file)
        preds, y = learn0.get_preds(dl=test_dl)

        predicted_mask = np.argmax(preds, axis=1)
        plt.imshow(predicted_mask[0])
        a = np.array(predicted_mask[0])

        tumor_p = False

        print('Current slice:', curr_slice)

        unique = np.unique(a)
        print("Predicted", unique)
        if 0 in unique:
            back_p = True
        else:
            back_p = False
        if 1 in unique:
            liver_p = True
        else:
            liver_p = False
        if 2 in unique:
            tumor_p = True
        else:
            tumor_p = False

        # For getting the actual mask values
        mask = Image.fromarray(curr_mask[..., curr_slice].astype('uint8'), mode="L")
        tumor_t = False

        unique = np.unique(mask)
        print("Actual:", unique)
        if 0 in unique:
            back_t = True
        else:
            back_t = False
        if 1 in unique:
            liver_t = True
        else:
            liver_t = False
        if 2 in unique:
            tumor_t = True
        else:
            tumor_t = False

        # Populating the confusion matrix
        if tumor_p == True and tumor_t == True:
            conf_matrix[0, 0] += 1
        if tumor_p == False and tumor_t == False:
            conf_matrix[1, 1] += 1
        if tumor_p == False and tumor_t == True:
            conf_matrix[1, 0] += 1
        if tumor_p == True and tumor_t == False:
            conf_matrix[0, 1] += 1

print(conf_matrix)

# Plot Confusion Matrix
import seaborn as sns

ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ')

# Set tick labels
ax.xaxis.set_ticklabels(['True', 'False'])
ax.yaxis.set_ticklabels(['True', 'False'])

# Display the visualization of the Confusion Matrix
plt.show()

# Converting .nii files to jpg and masks
nums = [3, 4, 5]
vol_names = ['volume-100.nii', 'volume-102.nii', 'volume-102.nii']
seg_names = ['segmentation-100.nii', 'segmentation-102.nii', 'segmentation-102.nii']

total_slice = 0

for nums, vol_names, seg_names in zip(nums, vol_names, seg_names):
    curr_ct = read_nii(df_files.loc[nums, 'dirname'] + "/" + df_files.loc[nums, 'filename'])
    curr_mask = read_nii(df_files.loc[nums, 'mask_dirname'] + "/" + df_files.loc[nums, 'mask_filename'])
    curr_file_name = str(df_files.loc[nums, 'filename']).split('.')[0]
    curr_dim = curr_ct.shape[2]

    for curr_slice in tqdm(range(250, 550, 10)):  # Export every 10th slice for testing
        data = tensor(curr_ct[..., curr_slice].astype(np.float32))
        mask = Image.fromarray(curr_mask[..., curr_slice].astype('uint8'), mode="L")
        data.save_jpg(f"images/{curr_file_name}_slice_{curr_slice}.jpg", [dicom_windows.liver, dicom_windows.custom])
        mask.save(f"mask/{curr_file_name}_slice_{curr_slice}_mask.png")
        total_slice = total_slice + 1

print(total_slice)

# Using OpenCV to find if the mask of a scan image is showing any of the features (background, liver, and tumor).
# The data is being saved into a CSV file. This can later be used to validate the model by comparing the model prediction with the CSV files.

import cv2
import numpy as np
import os

def check(img):
    cnt, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) > 0:
        return 1
    else:
        return 0

f = open('true_values.csv', mode='w')

file_iterator = os.scandir('./mask')
for i in file_iterator:
    img = cv2.imread("./mask/" + i.name, -1)
    print(img.shape)

    # If only the background is visible
    if np.count_nonzero(img) == 0:
        f.write('0,0,0\n')
    else:
        f.write('0,')
        # For liver
        img_liver = np.where(img == 1, 255, img)
        img_liver = np.where(img_liver == 2, 0, img)
        ret = check(img_liver)
        if ret:
            f.write('1,')
        else:
            f.write('0,')

        # For tumor
        img_tumor = np.where(img == 2, 255, img)
        img_tumor = np where(img_tumor == 1, 0, img)
        ret = check(img_tumor)
        if ret:
            f.write('1')
        else:
            f.write('0')

        f.write('\n')

f.close()
