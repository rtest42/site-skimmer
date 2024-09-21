import csv
import numpy as np
import os
import re
import requests
import sys
import threading
import time
import torch.cuda
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from glob import glob
from io import BytesIO
# For duplicate images - Keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
# Clothing segment
from PIL import Image
from transformers import pipeline

start_time = time.perf_counter()

# Prefer GPU (0) over CPU (-1)
device = 0 if torch.cuda.is_available() else -1
# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes", device=device)

# Get list of clothing categories
with open('categories.txt', 'r') as clothing_file:
    clothing_categories = clothing_file.readlines()

# Load the VGG16 model pre-trained on ImageNet
weights = 'imagenet'
if os.path.isfile('vgg16_weights_tf_dim_ordering_tf_kernels.h5'):
    weights = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
base_model = VGG16(weights=weights)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


# Returns a list of matching patterns
def get_files(patterns) -> list[str]:
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
    # Patch for files with bad file extension
    if image_url.find('?') >= 0:
        image_url = image_url[:image_url.find('?')]

    # Add missing HTTPS
    if not image_url.startswith('http'):
        image_url = 'https:' + image_url

    return image_url


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


# Sub-function for duplicate images: loading
# @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def preprocess_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data


# Sub-function for duplicate images: extracting
def extract_features(img_path_chunk) -> dict[int: np.array]:
    image_vectors = {}
    for img_path in img_path_chunk:
        img_data = preprocess_image(img_path)
        features = model.predict(img_data)
        # Get image number from path
        # Images may not be finished in order, so add a key
        image_key = re.search(r"\d+\.\D{3,4}$", img_path)
        image_key = image_key.group(0)
        image_key = image_key.partition('.')[0]
        image_key = int(image_key)
        print("Extracted image", image_key)
        image_vectors[image_key] = features

    return image_vectors


# Function for getting image vectors using multiprocessing
def extract_features_multiprocessing(image_names) -> dict[int: np.array]:
    # Split the list of images
    cores = os.cpu_count() or 1
    image_chunks = [image_names[i::cores] for i in range(cores)]
    features = {}
    # Test using threads instead of processes
    with ProcessPoolExecutor(cores) as executor:
        results = executor.map(extract_features, image_chunks)

    for result in results:
        # Merge dictionaries (3.9+)
        features |= result
        # features.update(result)

    return features


# Function to check image similarity
def cosine_similarity(features_a, features_b) -> float:
    dot_product = np.dot(features_a, features_b.T)
    norm_a = np.linalg.norm(features_a)
    norm_b = np.linalg.norm(features_b)
    try:
        return dot_product / (norm_a * norm_b)
    except ZeroDivisionError:
        return 1.0  # Undefined; flag as same


# Given a decimal convert to bitmap
def parse_seg_mode(value) -> list[str]:
    options = clothing_categories
    output = []
    value_binary = str(bin(value)[2:].zfill(len(options)))

    for i in range(len(value_binary)):
        if value_binary[i] == '1':
            output.append(options[i])

    return output


# Helper function for segmenting clothing
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


# Save segmented clothing
def batch_segment_clothing(filename, out_dir, clothes) -> int:
    if filename[filename.find('.'):] not in ['.jpg', '.JPG', '.png', '.PNG']:
        print("Skipping", filename, "as it is not a supported image file.")
        return 0

    # Iterate through each file in the input directory
    try:
        # Load image
        img = Image.open(filename).convert("RGBA")

        # Segment clothing
        segmented_img = segment_clothing(img, clothes)

        # Save segmented image to output directory as PNG
        out_path = os.path.join(out_dir, filename.partition('/')[2].partition('.')[0] + ".png")
        segmented_img.save(out_path)

        print("Segmented", filename, "successfully.")
        return 1  # Success in this case
    except IndexError as e:
        print("Error processing", filename, "-", e)
        return 0


# Segment multiple images
def batch_segment_clothing_multiprocessing(img_dir, out_dir, clothes=None) -> int:
    clothes = clothes or clothing_categories
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    cores = os.cpu_count() or 1

    with ProcessPoolExecutor(cores) as executor:
        results = {executor.submit(batch_segment_clothing, file, out_dir, clothes)
                   for file in glob(f"{img_dir}/*")}

    return sum(future.result() for future in results)


# Helper function for downloading images
def download_image(image_url, image_label, counter, counter_lock):
    # Make image URL valid
    image_url = check_image_link(image_url.get('src'))

    # Skip files with bad file extension (.svg)
    file_extension = os.path.splitext(image_url)[-1].lower()
    if file_extension.find('.svg') != -1:
        return None

    # Add missing file extension (.jpg for now)
    if file_extension.find('.') == -1:
        file_extension = '.jpg'

    # Download the image
    try:
        image_data = requests.get(image_url).content
    except requests.exceptions.InvalidURL as e:
        print(e)
        return None

    # Check image before saving to disk
    if not pre_check_image(image_data):
        return None

    # Name file
    # Lock thread to prevent race condition for counter
    with counter_lock:
        image_name = os.path.join(image_label, f"{image_label}{counter[0]}{file_extension}")
        print("Fetched image", counter[0])
        counter[0] += 1  # Safely update counter

    # Write image
    with open(image_name, 'wb') as f:
        f.write(image_data)

    # Metadata
    return {
        'img_url': image_url,
        'img_name': image_name,
        'img_label': image_label
    }


# Downloads images and its content
def download_images(images, image_label) -> dict:
    image_metadata = []
    google_vertex = []
    counter = [0]  # Make the counter mutable
    counter_lock = threading.Lock()  # Lock counter
    # Create a directory to save images
    os.makedirs(image_label, exist_ok=True)

    # Utilize threading to download multiple images at once
    with ThreadPoolExecutor(max_workers=len(images)) as executor:
        futures = {executor.submit(download_image, img, image_label, counter, counter_lock) for img in images}

    for future in futures:
        metadata = future.result()
        # Append metadata to CSV
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


# Helper function for opening and extracting HTML contents from a file
def extract_tags(file, tag='img') -> list:
    with open(file, 'r', encoding='utf8') as f:
        print("Opening", file)
        # Read and parse text
        data = f.read()
        parsed_data = BeautifulSoup(data, 'html.parser')
        # List all tags' properties
        return parsed_data.find_all(tag)


# Get a list of tags from multiple files
def extract_tags_multithreading(files) -> list:
    tags = []
    with ThreadPoolExecutor(max_workers=len(files)) as executor:
        # Open multiple files concurrently
        results = executor.map(extract_tags, files)

    for result in results:
        tags.extend(result)

    return tags


# Flag all duplicate images
def list_duplicates(image_vectors, num_images, threshold=0.80) -> set[int]:
    duplicates = set()
    for i in range(num_images):
        # Skip those already flagged
        if i in duplicates:
            continue

        for j in range(i + 1, num_images):
            # Skip those already flagged
            if j in duplicates:
                continue
            # Compare similarity
            similarity = cosine_similarity(image_vectors[i], image_vectors[j])
            if similarity > threshold:
                print("Duplicate detected: images", i, "and", j)
                duplicates.add(j)

    return duplicates


# Remove duplicate images
def remove_duplicates(duplicate_set, label) -> int:
    files_to_remove = [f for num in duplicate_set for f in glob(f"{label}/*{num}.*")]
    with ThreadPoolExecutor(max_workers=len(files_to_remove)) as executor:
        executor.map(os.remove, files_to_remove)

    return len(duplicate_set)


# The main function receives HTML files as arguments and a label as an input
# The output consists of an image folder and several CSV files
def main() -> int:
    # Check for arguments
    if len(sys.argv) < 3:
        print("Usage: python3 skimmer.py <segment_mode> <pattern1> [pattern2] ...")
        return 1

    # Get segmentation mode (number)
    # Common bitmap values:
    # 6 - shoes
    # 16 - dresses
    # 96 - skirt, pants, lower-body clothing
    # 128 - shirt, jacket, upper-body clothing
    try:
        seg_mode = parse_seg_mode(int(sys.argv[1]))
    except ValueError:
        print("segment_mode has to be a number.")
        return 1

    # Get list of files
    files = get_files(sys.argv[2:])
    if files is None:
        print("ERROR: No files found!")
        return 1

    # Output directory
    while True:
        label = input("Enter a label (q to quit): ")
        # Ensure user input is valid
        if not label:
            print("Label should not be empty!")
        elif label[-1].isdigit():
            print("Label should not end in a digit! (for counting purposes)")
        elif label.lower() in ['q', 'quit']:
            print("Exiting program.")
            return 1
        else:
            break

    # Get image tags from all inputted files
    print("Processing input files...")
    image_tags = extract_tags_multithreading(files)

    # Download the images
    print("Downloading images...")
    image_metadata = download_images(image_tags, label)
    image_counter = len(image_metadata['image_metadata'])

    # Load images onto image vectors
    print("Processing images...")
    image_names = [image_name['img_name'] for image_name in image_metadata['image_metadata']]
    image_vectors = extract_features_multiprocessing(image_names)

    # Select all images that are duplicates, then remove them
    print("Removing duplicates...")
    image_duplicates = list_duplicates(image_vectors, image_counter)
    num_removed = remove_duplicates(image_duplicates, label)
    print("Removed", num_removed, "files")

    # Segment clothing and put them into another folder
    # ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
    #    0          0             0        1        0        0         0             0          0
    print("Batch segment clothing...")
    num_batched = batch_segment_clothing_multiprocessing(label, f"{label}_f", seg_mode)
    print("Batched", num_batched, "items")

    # Put images into CSV file
    print("Saving to CSV...")
    save_to_csv(image_metadata['image_metadata'], f"items_metadata_{label}.csv")
    save_to_csv(image_metadata['google_vertex'], f"google_vertex_{label}.csv")

    new_files = glob(f"{label}_f/*")
    new_g_vertex = [{'img_dir': f"gs://cloud-ml-data/{new_file}", 'label': label} for new_file in new_files]
    save_to_csv(new_g_vertex, f"google_vertex_{label}_f.csv")

    # Indicate successful run
    print("Done.")
    return 0


if __name__ == '__main__':
    exit_code = main()  # Run program
    print("Program finished with exit code", exit_code)
    print("Time elapsed:", time.perf_counter() - start_time, "seconds")
