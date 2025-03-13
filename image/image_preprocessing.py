import os.path
import numpy as np
import torch
import nibabel as nib
import random
from scipy.ndimage import zoom
from image.affine import data_affine
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def process_nii_to_npy(image_path, output_path, Image_aug,
                       affine, affine_size,
                       affine_time,
                       target_depth=32,
                       target_shape=(256, 256)):
    """
    Process a .nii file and convert it to a .npy file with shape 1*target_depth*target_shape[0]*target_shape[1].

    Args:
        image_path (str): Path to the input .nii file.
        output_path (str): Path to save the processed .npy file.
        target_depth (int): Target depth for the 3D image.
        target_shape (tuple): Target height and width for the 3D slices (default is 256x256).

    Returns:
        str: Path to the saved .npy file.
    """
    # Step 1: Load the .nii file
    nii_image = nib.load(image_path)
    image_data = nii_image.get_fdata()  # Extract data as a numpy array
    print(f"Original shape: {image_data.shape}")

    if Image_aug == True:  # and len(image_data.shape) == 4:
        output_path = output_path + '_{}'.format(affine_time)
        random_sample = random.choice(range(image_data.shape[3]))
        image_data = image_data[:, :, :, random_sample]
        # Affine
        if affine == True:
            image_data = data_affine(image_data)
        print(image_data.shape)
    if Image_aug == False:  # and len(image_data.shape) == 4:
        image_data = image_data[:, :, :, 0]

    depth_original = image_data.shape[2]
    resize_factor_depth = target_depth / depth_original
    # Resize height and width to match target_shape
    resize_factors = (target_shape[0] / image_data.shape[0],
                      target_shape[1] / image_data.shape[1],
                      resize_factor_depth)
    image_resized = zoom(image_data, resize_factors, order=1)  # Linear interpolation
    print(f"Shape after resize: {image_resized.shape}")

    # Step 3: Normalize to 0-1
    image_min = image_resized.min()
    image_max = image_resized.max()
    image_normalized = (image_resized - image_min) / (image_max - image_min)
    image_normalized.fill(0)
    print(f"Data range after normalization: {image_normalized.min()} to {image_normalized.max()}")

    # Step 4: Add batch dimension
    image_final = image_normalized[np.newaxis, :, :, :]  # Add batch dimension
    image_final = np.transpose(image_final, (0, 3, 1, 2))  # Rearrange axes to 1×32×256×256
    print(f"Final shape: {image_final.shape}")

    # Step 5: Save as .npy
    np.save(output_path, image_final)
    print(f"Saved processed image to {output_path}")

    return output_path


def get_npy(item_dir_list, output_path,name, name_dir_list, Image_aug, affine_time):
    for im_path, name_img in zip(item_dir_list, name_dir_list):
        if name == name_img:
            if Image_aug == False:
                sax_img = name + '_sax.nii.gz'
            if Image_aug == True:
                choices = ['_sax.nii.gz']
                sax_img = name + random.choice(choices)
            image_path = os.path.join(im_path, sax_img)
            output_path = process_nii_to_npy(image_path, output_path, Image_aug,
                                             affine=True,
                                             affine_size=50,  # How many t in affine
                                             affine_time=affine_time,
                                             target_depth=32,
                                             target_shape=(256, 256))
            image_path = output_path
            return image_path