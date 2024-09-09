import csv
import os
import requests
import sys
from bs4 import BeautifulSoup
from glob import glob
from io import BytesIO
from PIL import Image
# For duplicate images
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import cv2
# For clothing segment


# Returns a list of matching patterns
def get_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob(pattern))
    return files

# Convert list to CSV file
def save_to_csv(items, filename):
    if not items:
        return
    keys = items[0].keys()
    with open(filename, 'a', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(items)

# Check the image url to ensure validity before fetching
def check_image_link(image_url):
    output_url = image_url
    # Patch for files with bad file extension
    if output_url.find('?') >= 0:
        output_url = image_url[:image_url.find('?')]

    # Add missing HTTPS
    if not output_url.startswith('http'):
        output_url = 'https:' + output_url

    return output_url

# Image Pre-check before saving to disk
def precheck_image(image):
    # Skip files less than 8kB
    if len(image) >= 8000:
        # Skip grayscale images
        img = Image.open(BytesIO(image))
        img = img.convert('RGB')
        is_grayscale = all(r == g == b for r, g, b in img.getdata())
        if not is_grayscale:
            return True
    return False

# Subfunctions for duplicate images
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features
def cosine_similarity(featuresA, featuresB):
    dot_product = np.dot(featuresA, featuresB.T)
    normA = np.linalg.norm(featuresA)
    normB = np.linalg.norm(featuresB)
    return dot_product / (normA * normB)

# Checks if two images are similar
def check_for_duplicates(base_image_path, comparison_image_path):
    # Load the VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    # Extract features from the two images
    featuresA = extract_features(base_image_path, model)
    featuresB = extract_features(comparison_image_path, model)

    # Compare the features using cosine similarity
    similarity_score = cosine_similarity(featuresA, featuresB)

    # Determine if the images are very similar
    threshold = 0.80
    if (similarity_score > threshold):
        return True
    return False

# Downloads images and its content
def download_images(html_text, image_label, counter=0):
    image_metadata = []
    google_vertex = []

    # Create a directory to save images
    os.makedirs(image_label.replace(' ', '_').lower(), exist_ok=True)

    images = html_text.find_all('img')
    for image in images:
        temp_image_url = image.get('src')
        if temp_image_url:
            image_url = check_image_link(temp_image_url)

            # Skip files with bad file extension (.svg)
            file_extension = os.path.splitext(image_url)[-1].lower()
            if file_extension.find('.svg') != -1:
                continue
            # Add missing file extension (.jpg for now)
            if file_extension.find('.') == -1:
                file_extension = '.jpg'

            # Download the image
            try:
                image_data = requests.get(image_url).content
            except requests.exceptions.InvalidURL as e:
                print("Invalid URL: ", e)
                continue

            # Verify image is not grayscale or less than 8kB
            if precheck_image(image_data):
                # Save image to folder
                image_name = os.path.join(image_label.replace(' ', '_').lower(), f"{image_label}{counter}{file_extension}")
                with open(image_name, 'wb') as f:
                    f.write(image_data)

                print("Wrote image " +str(counter))
                # Check if image is a duplicate (using five images before as comparison)
                if (counter > 5):
                    duplicate_found = False
                    for i in range(5):
                        image_to_check = image_label + str(counter - i) + file_extension

                        if check_for_duplicates(image_name, "./" + image_label + "/" + image_to_check):
                            duplicate_found = True
                            os.remove(image_name)
                            break

                    if not duplicate_found:
                        # Save item metadata
                        image_info = {
                            'img_url': image_url,
                            'img_name': image_name,
                            'img_label': image_label
                        }
                        image_metadata.append(image_info)

                        # Create file for Google Vertex
                        vertex = {
                            'img_dir': "gs://cloud-ml-data/{}".format(image_name),  # Change directory as necessary
                            'label': image_label
                        }
                        google_vertex.append(vertex)

                counter += 1

    return {
        'image_metadata': image_metadata,
        'google_vertex': google_vertex,
        'counter': counter
    }


# The main function receives HTML files as arguments and a label as an input
# The output consists of an image folder and several CSV files
def main():
    # Check for arguments
    if len(sys.argv) < 2:
        print("Usage: python3 skimmer.py <pattern1> [pattern2] ...")
        sys.exit(1)

    # Get list of files
    args = sys.argv[1:]
    files = get_files(args)

    if files is None:
        print("No files found!")
        sys.exit(0)

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
            metadata = download_images(parsed_data, label, image_count)

            # Update counter
            image_count = metadata['counter']
            # Put images into CSV file
            save_to_csv(metadata['image_metadata'], 'items_metadata_{}.csv'.format(label.replace(' ', '_').lower()))
            save_to_csv(metadata['google_vertex'], 'google_vertex_{}.csv'.format(label.replace(' ', '_').lower()))


if __name__ == "__main__":
    main()
