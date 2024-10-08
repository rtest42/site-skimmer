import os
import sys
import time
import torch
from download_images import download_images, extract_tags
from generate_csv import generate_csv
from glob import glob
from image_segmentation import parse_segmentation_mode, segment_images
from PIL import ImageFile
from remove_duplicates import extract_features, list_duplicates, remove_duplicates
from torchvision import models
from transformers import pipeline


def main(args=sys.argv) -> None:
    if len(args) < 2:
        print("Usage: python3 skimmer.py <pattern1> [pattern2] ...")
        return
    
    # Variables for downloading images
    files = []
    patterns = args[1:]
    for pattern in patterns:
        files.extend(glob(pattern))

    if len(files) == 0:
        print("No files found!")
        return
    
    segmentation = int(input("Enter a segmentation mode (integer): "))
    label = input("Enter a label: ")
    start = time.perf_counter()

    # Download images from HTML files
    tags = extract_tags(files)
    if len(tags) == 0:
        print("No images found!")
        return
    
    download_images(tags, label)

    # Remove duplicate images
    input_directory = label
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.isfile("vgg16-397923af.pth"):
        model = models.vgg16().to(device) 
        model.load_state_dict(torch.load("vgg16-397923af.pth"))
    else:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
        torch.save(model, "vgg16-397923af.pth")
    model.eval()

    # Remove duplicate images
    features = extract_features(input_directory, model)
    files = list_duplicates(input_directory, features)
    remove_duplicates(list(files))

    # Variables for segmenting images
    categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
    segmentations = parse_segmentation_mode(segmentation, categories)
    pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0 if torch.cuda.is_available() else -1)
    output_directory = input_directory + "-f"

    # Segment images
    segment_images(input_directory, output_directory, segmentations, pipe)

    # Generate CSV
    generate_csv(output_directory, label)
    
    # Finish program
    end = time.perf_counter()
    print(f"Program finished in {end - start} sesconds")


if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    main()