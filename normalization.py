import torchstain
import numpy as np
import cv2
import os
from tqdm import tqdm

# Function to find median stain_matrix for normalization
def normalizer_fit(filenames: list[str], normalizer: torchstain.normalizers.MacenkoNormalizer, size=224):

    stain_matrices = []
    maxCRefs = []

    for file in tqdm(filenames, desc="Fitting normalizer"):
        target = cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), (size, size))

        # fit the normalizer with the image
        normalizer.fit(target)

        stain_matrices.append(normalizer.HERef)
        maxCRefs.append(normalizer.maxCRef)

    normalizer.HERef = np.median(stain_matrices, axis=0)
    normalizer.maxCRef = np.median(maxCRefs, axis=0)

def normalize_dataset(data_dir: str, size=224):
    # Get the class names from the data directorys subdirectories
    class_names = os.listdir(data_dir)
    class_names = [class_name for class_name in class_names if os.path.isdir(os.path.join(data_dir, class_name))]

    # Check if normalized_data exists in ../data directory and create it if it doesn't
    if not os.path.exists(os.path.join(data_dir, '../normalized_data')):
        os.mkdir(os.path.join(data_dir, '../normalized_data'))
    normalized_path = os.path.join(data_dir, '../normalized_data')
    for class_name in class_names:
        if not os.path.exists(os.path.join(normalized_path, class_name)):
            os.mkdir(os.path.join(normalized_path, class_name))
    
    # Create the normalizer
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    
    images_list=[]
    # Get the list of images in data directory
    for class_name in class_names:
        image_files = os.listdir(os.path.join(data_dir, class_name))
        image_files = [image_file for image_file in image_files if image_file.endswith('.jpg')]
        images_list.extend([os.path.join(data_dir, class_name, image_file) for image_file in image_files])
    
    # Fit the normalizer with images
    normalizer_fit(images_list, normalizer,size)

    # Normalize all the images
    for class_name in tqdm(class_names, desc="Normalizing images"):
        image_files = os.listdir(os.path.join(data_dir, class_name))
        image_files = [image_file for image_file in image_files if image_file.endswith('.jpg')]
        image_files = [os.path.join(data_dir, class_name, image_file) for image_file in image_files]
        for image_file in image_files:
            # Ceck if it exists in normalized_data
            if os.path.exists(os.path.join(normalized_path, class_name, image_file.split('/')[-1])):
                continue
            # Read the image
            image = cv2.resize(cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB), (size, size))
            # Normalize the image
            normalized_image,_,_ = normalizer.normalize(I=image)
            # Save the image
            cv2.imwrite(os.path.join(normalized_path, class_name, os.path.basename(image_file)), cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))




        

