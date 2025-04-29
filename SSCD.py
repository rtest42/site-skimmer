import json
import torch
import os
import numpy as np
import subprocess
from PIL import ImageFile, Image
from image_segmentation import load_images
from datasets import load_dataset
from download_images import download_images
from transformers import pipeline

from pathlib import Path
from skimmer3 import skimming, detection, main
from tqdm import tqdm

class SSCD(object):
    def __init__(self, label):
        self.label = label
        self.categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
        self.pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0 if torch.cuda.is_available() else -1)

        
    def load_dataset(self) -> None:
        self.dataset = load_dataset("imagefolder", data_dir=self.label, split='test', keep_in_memory=True)


    def skimmer(self, searches: list[str], folders: list[str], rounds: int) -> None:
        for search, folder in zip(searches, folders):
            print(f"Searching {search} via Pinterest")
            os.makedirs(os.path.join(self.label, 'test', folder), exist_ok=True)
            result = subprocess.run(["node", "puppeteer-main.js", search, str(rounds)], capture_output=True, text=True)
            result = json.loads(result.stdout.strip())
            filtered_results = [url for url in result if '236x' in url] # Only keep images with a width of 236px
            print(filtered_results)
            download_images(filtered_results, os.path.join(self.label, 'test', folder), checks=False) # TODO: make method cleaner somehow

    
    def detection(self):
        pass

    def clipping(self, output_directory: str = "label1output") -> None:
        os.makedirs(output_directory, exist_ok=True)
        # segments = self.pipe(self.dataset)

        label_names = self.dataset.features['label'].names
        
        for i, data in enumerate(tqdm(self.dataset, desc="piping")):
            image = data['image']
            file = image.filename

            output = self.pipe(image)

            # Extract the label
            label_id = data['label']  # This gives the numeric ID of the label
            label_name = label_names[label_id]  # Convert ID to class name

            # Append all masks to output
            masks = []
            labels = []
            for segment in output:
                label = segment['label']
                labels.append(label)
                if label in self.categories:
                    masks.append(segment['mask'])
            
            # Logic for handling masks
            # Detection
            category = 'bad'
            if 'Face' in labels and ('Left-shoe' in labels or 'Right-shoe' in labels):
                category = 'good'

            os.makedirs(os.path.join(category, label_name), exist_ok=True)
            image.save(os.path.join(category, label_name, os.path.basename(file)))

            # Clipping (only for Label 1)
            if self.label == '1':
                if len(masks) == 0:
                    print("Skipping-no masks detected")
                    continue

                stack = np.array(masks[0])
                for mask in masks:
                    stack += np.array(mask)

                image.putalpha(Image.fromarray(stack))
                os.makedirs(os.path.join(output_directory, label_name), exist_ok=True)
                image.save(os.path.join(output_directory, label_name, os.path.basename(file)))


def main() -> None:
    # Input for AI Label One or Two
    ai_label = input("AI Label 1 or 2? (1/2): ").strip()
    if ai_label not in ('1', '2'):
        print("Bad input")
        exit(1)

    method = input("Perform skimming in addition to detection (and clipping for Label 1)? (y/n): ").strip().lower()
    sscd = SSCD(f"label{ai_label}")

    # PERFORM SKIMMER
    if method in ('y', 'yes'):
        search = input("Enter your search terms (separate by comma): ").split(',')
        rounds = int(input("Enter the number of rounds for sifting: "))
        folder = input("Enter the folder names (separate by comma; leave blank for the same as the search terms): ").split(',')
        # Format folder output
        if len(folder) > len(search):
            folder = search
        elif len(folder) < len(search):
            for i in range(len(search)):
                if i >= len(folder):
                    folder.append(search[i])

        sscd.skimmer(search, search, rounds)

    sscd.load_dataset()
    # PERFORM DETECTION AND/OR CLIPPING
    sscd.clipping("label1background")


if __name__ == '__main__':
    main()
