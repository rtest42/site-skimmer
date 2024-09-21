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
device = {
    'hugging_face': 0 if torch.cuda.is_available() else -1,
    'torch': 'cuda' if torch.cuda.is_available() else 'cpu'
}
# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes", device=device['hugging_face'])

# Get list of clothing categories
with open('categories.txt', 'r') as clothing_file:
    clothing_categories = clothing_file.readlines()

# Load the VGG16 model pre-trained on ImageNet
model_name = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
weights = model_name if os.path.isfile(model_name) else 'imagenet'
base_model = VGG16(weights=weights)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


# Returns a list of matching patterns
def get_files(patterns) -> list[str]:
    return [file for pattern in patterns for file in glob(pattern)]


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
    image_url = image_url.partition('?')[0]  # Patch for files with bad file extension
    return image_url if image_url.startswith('http') else 'https:' + image_url  # Add missing HTTPS if needed


# Image pre-check before saving to disk
def pre_check_image(image) -> bool:
    # Skip files less than 8kB
    if len(image) < 8000:
        return False

    # Skip grayscale images
    with Image.open(BytesIO(image)) as img:
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
    img_key_pattern = re.compile(r"\d+\.\D{3,4}$")
    image_vectors = {}

    with ThreadPoolExecutor() as executor:
        img_data_list = list(executor.map(preprocess_image, img_path_chunk))

    features = model.predict(np.vstack(img_data_list))  # Batch prediction
    # Get image number from path
    # Images may not be finished in order, so add a key
    for idx, img_path in enumerate(img_path_chunk):
        image_key = int(re.search(img_key_pattern, img_path).group(0).partition('.')[0])
        print(f"Extracted image {image_key}")
        image_vectors[image_key] = features[idx]

    return image_vectors


# Function for getting image vectors using multiprocessing
def extract_features_multiprocessing(image_names) -> dict[int: np.array]:
    # Split the list of images
    cores = os.cpu_count() or 1
    image_chunks = [image_names[i::cores] for i in range(cores)]
    features = {}
    # Test using threads instead of processes
    with ProcessPoolExecutor() as executor:
        results = executor.map(extract_features, image_chunks)

    for result in results:
        # Merge dictionaries
        features.update(result)

    return features


# Function to check image similarity
def cosine_similarity(features_a, features_b) -> float:
    features_a = torch.tensor(features_a).to(device['torch'])
    features_b = torch.tensor(features_b).to(device['torch'])
    return torch.nn.functional.cosine_similarity(features_a, features_b).item()


# Given a decimal convert to bitmap
def parse_seg_mode(value) -> list[str]:
    return [category.strip() for category, is_selected in
            zip(clothing_categories, bin(value)[2:].zfill(len(clothing_categories))) if is_selected == '1']


# Helper function for segmenting clothing
def segment_clothing(img, clothes=None):
    clothes = clothes or clothing_categories
    segments = segmenter(img)  # Segment image
    mask_list = [seg['mask'] for seg in segments if seg['label'] in clothes]  # Create list of masks
    final_mask = np.sum(np.stack(mask_list), axis=0)  # Paste all masks on top of each other
    final_mask = Image.fromarray(final_mask)  # Convert final mask from np array to PIL image
    img.putalpha(final_mask)  # Apply mask to original image
    return img


# Save segmented clothing
def batch_segment_clothing(filename, out_dir, clothes) -> int:
    if not filename.lower().endswith(('.jpg', '.png')):
        print(f"Skipping {filename} as it is not a supported image file.")
        return 0

    # Iterate through each file in the input directory
    try:
        # Load image
        with Image.open(filename) as img:
            # Segment clothing
            img = img.convert("RGBA")
            segmented_img = segment_clothing(img, clothes)

        # Save segmented image to output directory as PNG
        out_path = os.path.join(out_dir, filename.partition('/')[2].partition('.')[0] + ".png")
        segmented_img.save(out_path)
        print(f"Segmented {filename} successfully.")
        return 1  # Success in this case
    except IndexError as e:
        print(f"Error processing {filename}: {e}")
        return 0


# Segment multiple images
def batch_segment_clothing_multiprocessing(img_dir, out_dir, clothes=None) -> int:
    clothes = clothes or clothing_categories
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        results = {executor.submit(batch_segment_clothing, file, out_dir, clothes)
                   for file in glob(f"{img_dir}/*")}

    return sum(future.result() for future in results)


# Helper function for downloading images
def download_image(session, image_url, image_label, counter, counter_lock):
    # Make image URL valid
    image_url = check_image_link(image_url.get('src'))

    # Skip files with bad file extension (.svg)
    file_extension = os.path.splitext(image_url)[-1].lower()
    if '.svg' in file_extension:
        return None

    # Add missing file extension (.jpg for now)
    file_extension = file_extension if '.' in file_extension else '.jpg'

    # Download the image
    try:
        image_data = session.get(image_url).content
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
        print(f"Fetched image {counter[0]}")
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
    session = requests.Session()
    # Create a directory to save images
    os.makedirs(image_label, exist_ok=True)

    # Utilize threading to download multiple images at once
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_image, session, img, image_label, counter, counter_lock) for img in images}

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
        print(f"Opening {file}")
        # Read and parse text
        parsed_data = BeautifulSoup(f.read(), 'html.parser')
        # List all tags' properties
        return parsed_data.find_all(tag)


# Get a list of tags from multiple files
def extract_tags_multithreading(files) -> list:
    with ThreadPoolExecutor() as executor:
        # Open multiple files concurrently
        results = executor.map(extract_tags, files)

    return [tag for result in results for tag in result]


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
            if cosine_similarity(image_vectors[i], image_vectors[j]) > threshold:
                print(f"Duplicate detected: images {i} and {j}")
                duplicates.add(j)

    return duplicates


# Remove duplicate images
def remove_duplicates(duplicate_set, label) -> int:
    files_to_remove = [f for num in duplicate_set for f in glob(f"{label}/*{num}.*")]
    with ThreadPoolExecutor() as executor:
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
    print(f"Removed {num_removed} files")

    # Segment clothing and put them into another folder
    # ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
    #    0          0             0        1        0        0         0             0          0
    print("Batch segment clothing...")
    num_batched = batch_segment_clothing_multiprocessing(label, f"{label}_f", seg_mode)
    print(f"Batched {num_batched} items")

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
    print(f"Program finished with exit code {exit_code}")
    print(f"Time elapsed: {time.perf_counter() - start_time} seconds")
