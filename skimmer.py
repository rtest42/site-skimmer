import csv
# import cv2  # Duplicate images
import itertools
import numpy as np  # Duplicate images
import os
import requests
import sys
import time
import torch.cuda
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor  # Threading
from glob import glob
from io import BytesIO
# For duplicate images
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
# Clothing segment
from PIL import Image
from transformers import pipeline

start_time = time.time()

# Prefer GPU (0) over CPU (-1)
device = 0 if torch.cuda.is_available() else -1
# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes", device=device)

# Get list of clothing categories
with open('categories.txt', 'r') as f:
    clothing_categories = f.readlines()

# Load the VGG16 model pre-trained on ImageNet
if os.path.isfile('vgg16_weights_tf_dim_ordering_tf_kernels.h5'):
    base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5')
else:
    base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


# Returns a list of matching patterns
def get_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob(pattern))
    return files


# Convert list to CSV file
def save_to_csv(items, filename) -> None:
    if not items:
        return
    keys = items[0].keys()
    with open(filename, 'a', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(items)


# Check the image url to ensure validity before fetching
def check_image_link(image_url) -> str:
    output_url = image_url
    # Patch for files with bad file extension
    if output_url.find('?') >= 0:
        output_url = image_url[:image_url.find('?')]

    # Add missing HTTPS
    if not output_url.startswith('http'):
        output_url = 'https:' + output_url

    return output_url


# Image pre-check before saving to disk
def pre_check_image(image) -> bool:
    # Skip files less than 8kB
    if len(image) < 8000:
        return False

    # Skip grayscale images
    img = Image.open(BytesIO(image))
    img = img.convert('RGB')
    is_grayscale = all(r == g == b for r, g, b in img.getdata())
    return not is_grayscale


# Sub-functions for duplicate images
def extract_features(img_path, image_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = image_model.predict(img_data)
    return features


# Function to check image similarity
def cosine_similarity(features_a, features_b):
    dot_product = np.dot(features_a, features_b.T)
    norm_a = np.linalg.norm(features_a)
    norm_b = np.linalg.norm(features_b)
    return dot_product / (norm_a * norm_b)


# Given a decimal convert to bitmap
def parse_seg_mode(value):
    options = clothing_categories
    output = []
    value_binary = str(bin(value)[2:].zfill(len(options)))

    for i in range(len(value_binary)):
        if value_binary[i] == '1':
            output.append(options[i])

    return output


def segment_clothing(img, clothes=None):
    clothes = clothes or clothing_categories
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    mask_list = []
    for seg in segments:
        if seg['label'] in clothes:
            mask_list.append(seg['mask'])

    # Paste all masks on top of each other
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask

    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask)

    # Apply mask to original image
    img.putalpha(final_mask)

    return img


def batch_segment_clothing(img_dir, out_dir, clothes=None):
    clothes = clothes or clothing_categories
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Iterate through each file in the input directory
    for filename in os.listdir(img_dir):
        if filename[filename.find('.'):] in ['.jpg', '.JPG', '.png', '.PNG']:
            try:
                # Load image
                img_path = os.path.join(img_dir, filename)
                img = Image.open(img_path).convert("RGBA")

                # Segment clothing
                segmented_img = segment_clothing(img, clothes)

                # Save segmented image to output directory as PNG
                out_path = os.path.join(out_dir, filename.split('.')[0] + ".png")
                segmented_img.save(out_path)

                print(f"Segmented {filename} successfully.")
            except IndexError as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipping {filename} as it is not a supported image file.")


# Helper function for downloading images
def download_image(image_url, image_label, counter, image_model, image_vectors):
    # Make image URL valid
    image_url = check_image_link(image_url)

    # Skip files with bad file extension (.svg)
    file_extension = os.path.splitext(image_url)[-1].lower()
    if file_extension.find('.svg') != -1:
        return

    # Add missing file extension (.jpg for now)
    if file_extension.find('.') == -1:
        file_extension = '.jpg'

    # Download the image
    try:
        image_data = requests.get(image_url).content
    except requests.exceptions.InvalidURL as e:
        print(e)
        return

    if not pre_check_image(image_data):
        return

    num = next(counter)
    image_name = os.path.join(image_label, f"{image_label}{num}{file_extension}")
    with open(image_name, 'wb') as f:
        f.write(image_data)

    # Get image vector
    feature = extract_features(image_name, image_model)
    # Check if image is a duplicate (using five images before as comparison)
    # for image_vector in image_vectors:  # [-5:]:
    #    if cosine_similarity(feature, image_vector) > 0.8:
    #        print(f"Found duplicate image {counter}")
    #        os.remove(image_name)
    #        return

    # Unique image
    image_vectors.append(feature)
    print(f"Fetched image {num}")

    return {
        'img_url': image_url,
        'img_name': image_name,
        'img_label': image_label
    }


# Downloads images and its content
def download_images(html_text, image_label, image_model):
    image_vectors = []
    image_metadata = []
    google_vertex = []
    counter = itertools.count(0)

    # Create a directory to save images
    os.makedirs(image_label, exist_ok=True)

    images = html_text.find_all('img')
    # Utilize threading to download multiple images at once
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(download_image, img.get('src'), image_label, counter, model, image_vectors) for img in images}

    print(len(image_vectors))
    # Get data for CSV files
    for future in future_to_image:
        metadata = future.result()
        if metadata:
            image_metadata.append(metadata)
            google_vertex.append({
                'img_dir': f"gs://cloud-ml-data/{metadata['img_name']}",  # Change directory as necessary
                'label': metadata['img_label']
            })

    return {
        'image_metadata': image_metadata,
        'google_vertex': google_vertex
    }


# The main function receives HTML files as arguments and a label as an input
# The output consists of an image folder and several CSV files
def main():
    # Check for arguments
    if len(sys.argv) < 3:
        print("Usage: python3 skimmer.py <segment_mode> <pattern1> [pattern2] ...")
        sys.exit(1)

    # Get list of files
    seg_mode = parse_seg_mode(int(sys.argv[1]))
    files = get_files(sys.argv[2:])

    if files is None:
        print("ERROR: No files found!")
        sys.exit(1)

    label = input("Enter a label: ")
    image_count = 0

    # Open and read all the text files
    for file in files:
        with open(file, 'r', encoding="utf8") as f:
            print(f"Opening {file}")

            # Read and parse text
            data = f.read()
            parsed_data = BeautifulSoup(data, 'html.parser')

            # Fetch metadata
            metadata = download_images(parsed_data, label, model)
            # Create a directory to save images
            label_name = label
            os.makedirs(label_name, exist_ok=True)
            input_dir = label_name
            os.makedirs(label_name + "_f", exist_ok=True)
            output_dir = label_name + "_f"
            # ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
            #    0          0             0        1        0        0         0             0          0
            batch_segment_clothing(input_dir, output_dir, clothes=seg_mode)

            # Update counter
            # image_count = metadata['counter']
            # Put images into CSV file
            save_to_csv(metadata['image_metadata'], f"items_metadata_{label}.csv")
            save_to_csv(metadata['google_vertex'], f"google_vertex_{label}.csv")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Done. Time elapsed: ", time.time() - start_time, " seconds")
