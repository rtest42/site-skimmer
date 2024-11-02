import numpy as np
import os
import sys
import torch
from glob import glob
from PIL import Image, ImageFile
from transformers import pipeline


# Convert integer to a list of segmentation modes
# ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
def parse_segmentation_mode(value: int, options: list[str]) -> list[str]:
    categories = []
    value_binary = str(bin(value)[2:].zfill(len(options)))

    for i in range(len(value_binary)):
        if value_binary[i] == '1':
            categories.append(options[i])
    
    return categories


# Load images
def load_images(directory: str) -> tuple[list[Image.Image], list[str]]:
    files = glob(f"{directory}/*")
    images = []
    for file in files:
        with Image.open(file) as image:
            image = image.convert("RGBA")
            images.append(image.copy())

    return images, files


# Take important part of image and save it
# NOTE: Batching is not used in the pipeline
def segment_images(input_directory: str, output_directory: str, categories: list[str], pipe) -> None:
    os.makedirs(output_directory, exist_ok=True)
    images, files = load_images(input_directory)
    segments = pipe(images)

    for image, file, output in zip(images, files, segments):
        masks = []
        for segment in output:
            if segment['label'] in categories:
                masks.append(segment['mask'])

        if len(masks) == 0:
            print(f"Unable to segment {file}")
            continue
        
        stack = np.array(masks[0])
        for mask in masks:
            stack += np.array(mask)

        image.putalpha(Image.fromarray(stack))
        path = os.path.join(output_directory, os.path.basename(file).partition('.')[0] + ".png")
        image.save(path, "PNG")
        print(f"Segmented {file} successfully")


# Determine if masks exist
def extract_masks(input_directory: str, categories: list[str], pipe) -> None:
    images, files = load_images(input_directory)
    segments = pipe(images)

    for image, file, output in zip(images, files, segments):
        for segment in output:
            if segment['label'] not in categories:
                print(f'WARNING image {file} is bad')


# For debugging
def main(args) -> None:
    if len(args) < 3:
        print("Usage: python3 image_segmentation.py <segmentation_mode> <input-directory> [output-directory]")
        print("The recommended segmentation modes are:")
        print("6 - shoes")
        print("16 - dresses")
        print("96 - skirt, pants, lower-body clothing")
        print("128 - shirt, jacket, upper-body clothing")
        return
    
    categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
    segmentations = parse_segmentation_mode(int(args[1]), categories)
    pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0 if torch.cuda.is_available() else -1)

    segment_images(args[2], args[2] + "-f" if len(args) == 3 else args[3], segmentations, pipe)
    

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    main(sys.argv)
