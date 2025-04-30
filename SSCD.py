import logging
import json
import torch
import threading
import os
import sys
import numpy as np
import requests
import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor
from PIL import Image
from image_segmentation import load_images
from datasets import load_dataset, Image as dataset_image # Avoid confusion between PIL.Image
from transformers.pipelines.pt_utils import KeyDataset
from download_images import download_images
from transformers import pipeline

from pathlib import Path
from skimmer3 import skimming, detection, main
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SSCD(object):
    def __init__(self, label):
        self.label = label
        self.categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
        self.pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0 if torch.cuda.is_available() else -1)
        self.dataset = None

        
    def load_dataset(self) -> None:
        self.dataset = load_dataset("imagefolder", data_dir=self.label, split='test')
        # Workaround to keep file paths
        self.dataset = self.dataset.cast_column("image", dataset_image(decode=False)) # Prevent decoding, keep file paths
        labelnames = self.dataset.features['label'].names
        self.dataset = self.dataset.map(lambda entry: {**entry, 'filename': entry['image'], 'labelname': labelnames[entry['label']]}) # Map two more columns based on values in other columns
        self.dataset = self.dataset.cast_column("image", dataset_image(decode=True)) # Re-encode to PIL.Image; file path is kept in filename

    
    def download_image(self, session: requests.Session, url: str, label: str, counter: list[int], lock: threading.Lock) -> None:
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
        }
        timeout = 10 # seconds
        _, ext = os.path.splitext(url)
        counter_digits = len(str(counter[1]))

        try:
            response = session.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                with lock:
                    name = os.path.join(label, f"{os.path.basename(label)}{counter[0]:0{counter_digits}}{ext}")
                    counter[0] += 1

                with open(name, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Status code for {url}: {response.status_code}") # TODO convert to logging
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}") # TODO convert to logging


    def download_images(self, links: list[str], label: str) -> None:
        os.makedirs(label, exist_ok=True)
        session = requests.Session()
        counter = [0, len(links)] # [counter, number of links (constant)]
        lock = threading.Lock()

        with tqdm(total=len(links), desc="Downloading images") as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.download_image, session, link, label, counter, lock) for link in links]
                for _ in as_completed(futures):
                    pbar.update(1)


    def skimmer(self, searches: list[str], folders: list[str], rounds: int) -> None:
        # Skimming is slow on purpose to prevent anti-bot detection
        for search, folder in zip(searches, folders):
            print(f"Searching {search} via Pinterest")
            result = subprocess.run(["node", "puppeteer-main.js", search, str(rounds)], capture_output=True, text=True)
            result = json.loads(result.stdout.strip())
            filtered_results = [url for url in result if '236x' in url] # Only keep images with a width of 236px
            self.download_images(filtered_results, os.path.join(self.label, 'test', folder))

    
    def detection(self) -> None: # TODO: test - remove later
        for _, x, _ in tqdm(zip(self.pipe(KeyDataset(self.dataset, "image")), KeyDataset(self.dataset, "filename"), KeyDataset(self.dataset, "labelname")), total=len(self.dataset), desc="Segmenting images"):
            x['path']
            pass


    def image_segmentation(self, output_directory: str = "label1output") -> None:
        os.makedirs(output_directory, exist_ok=True)
        images = KeyDataset(self.dataset, "image")
        filenames = KeyDataset(self.dataset, "filename")
        labelnames = KeyDataset(self.dataset, "labelname")

        for image, filename, labelname in tqdm(zip(self.pipe(images), filenames, labelnames), total=len(self.dataset), desc="Segmenting images"):
            output = image

            # Append specified masks to output
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

            os.makedirs(os.path.join(category, labelname), exist_ok=True)
            image.save(os.path.join(category, labelname, os.path.basename(filename)))

            # Clipping (only for Label 1)
            if self.label == '1':
                if len(masks) == 0:
                    print("Skipping-no masks detected")
                    continue

                stack = np.array(masks[0])
                for mask in masks:
                    stack += np.array(mask)

                image.putalpha(Image.fromarray(stack))
                os.makedirs(os.path.join(output_directory, labelname), exist_ok=True)
                image.save(os.path.join(output_directory, labelname, os.path.basename(filename)))


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
        if len(folder) != len(search):
            folder = search

        sscd.skimmer(search, folder, rounds)

    sscd.load_dataset()
    # PERFORM DETECTION AND/OR CLIPPING
    # sscd.clipping("label1background")
    sscd.detection()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        stream=sys.stderr
    )

    main()
