import torch
import os
import numpy as np
from PIL import ImageFile
from datasets import load_dataset
from transformers import pipeline

if __name__ == '__main__':
    # ImageFile.LOAD_TRUNCATED_IMAGES = True
    # Variables for segmenting images
    categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
    # segmentations = parse_segmentation_mode(segmentation, categories)
    segmentations = [categories[1], categories[3]]
    pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0 if torch.cuda.is_available() else -1)
    # output_directory = input_directory + "-f"

    input_directory = 'label1-archive'

    dataset = load_dataset("imagefolder", data_dir=input_directory, split='test')

    # Get the label names (mapping from label IDs to class names)
    label_names = dataset.features['label'].names
    # Segment images
    # segment_images(input_directory, output_directory, segmentations, pipe)
    for data in dataset:
        image = data['image']
        file = image.filename
        output = pipe(image)
        pixels = {
            "Background": 0,
            "Hat": 0,
            "Hair": 0,
            "Sunglasses": 0,
            "Upper-clothes": 0,
            "Skirt": 0,
            "Pants": 0,
            "Dress": 0,
            "Belt": 0,
            "Left-shoe": 0,
            "Right-shoe": 0,
            "Face": 0,
            "Left-leg": 0,
            "Right-leg": 0,
            "Left-arm": 0,
            "Right-arm": 0,
            "Bag": 0,
            "Scarf": 0
        }
        width, height = image.size
        total_pixels = width * height

        # Extract the label
        label_id = data['label']  # This gives the numeric ID of the label
        label_name = label_names[label_id]  # Convert ID to class name

        for segment in output:
            mask = np.array(segment['mask'])
            mask = (mask > 127).astype(np.uint8)

            label = segment['label']
            # print(label, mask.sum(), total_pixels, mask)
            pixels[label] = mask.sum()
            pixels[label] = pixels[label] / total_pixels
            # pixels[label] = pixels[label] >= threshold

        if pixels['Face'] and (pixels['Left-shoe'] or pixels['Right-shoe']):
            category = "good"
            # Some more conditions here
        else:
            category = "bad"

        image.save(os.path.join(category, label_name, os.path.basename(file)))
        print(f'{os.path.basename(file)} categorized as {category}')