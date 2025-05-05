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
from datasets import Image as DatasetImage, load_dataset  # Avoid confusion between PIL.Image
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SSCD(object):
    def __init__(self, label: str):
        self.label = label
        self.categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress",
                           "Belt", "Left-shoe", "Right-shoe", "Scarf"]
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=device)
        self.dataset = None

    def load_dataset(self):
        self.dataset = load_dataset("imagefolder", data_dir=self.label, split='test')
        # Workaround to keep file paths
        # Prevent decoding, keep file paths
        self.dataset = self.dataset.cast_column("image", DatasetImage(decode=False))
        # Map two more columns based on values in other columns
        label_names = self.dataset.features['label'].names
        self.dataset = self.dataset.map(lambda entry: {**entry, 'filename': entry['image'],
                                                       'label-name': label_names[entry['label']]})
        # Re-encode to PIL.Image; file path is kept in filename
        self.dataset = self.dataset.cast_column("image", DatasetImage(decode=True))

    @staticmethod
    def download_image(session: requests.Session, url: str, label: str, counter: list[int], lock: threading.Lock):
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0'
        }
        timeout = 10  # seconds
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

                print(f"Successfully downloaded and saved {url}")  # TODO convert to logging
            else:
                print(f"Status code for {url}: {response.status_code}")  # TODO convert to logging
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")  # TODO convert to logging

    def download_images(self, links: list[str], label: str):
        os.makedirs(label, exist_ok=True)
        session = requests.Session()
        counter = [0, len(links)]  # [counter, number of links (constant)]
        lock = threading.Lock()

        with tqdm(total=len(links), desc="Downloading images") as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.download_image, session, link, label, counter, lock) for link in links]
                for _ in as_completed(futures):
                    # result = future.result()
                    pbar.update(1)

    def skimmer(self, searches: list[str], folders: list[str], rounds: int):
        # Skimming is slow on purpose to prevent anti-bot detection
        for search, folder in zip(searches, folders):
            print(f"Searching {search} via Pinterest")
            result = subprocess.run(["node", "puppeteer-main.js", search, str(rounds)], capture_output=True, text=True)
            result = json.loads(result.stdout.strip())
            filtered_results = [url for url in result if '236x' in url]  # Only keep images with a width of 236px
            self.download_images(filtered_results, os.path.join(self.label, 'test', folder))

    def image_segmentation(self):
        # Make directories
        for label_name in self.dataset.features['label'].names:
            for category in ["bad", "good", "output"]:
                os.makedirs(os.path.join(self.label, category, label_name), exist_ok=True)

        images = KeyDataset(self.dataset, "image")

        for i, image in tqdm(enumerate(self.pipe(images)), total=len(self.dataset), desc="Segmenting images"):
            filename = self.dataset[i]['filename']['path']
            label_name = self.dataset[i]['label-name']
            filename = os.path.basename(filename)

            # TODO work on logic below until end of loop/function
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

            image.save(os.path.join(category, label_name, filename))

            # Clipping (only for Label 1)
            if self.label == '1':
                if len(masks) == 0:
                    print("Skipping-no masks detected")  # TODO change to log
                    continue

                stack = np.array(masks[0])
                for mask in masks:
                    stack += np.array(mask)

                image.putalpha(Image.fromarray(stack))
                image.save(os.path.join('output', label_name, filename))


def main():
    # Input for AI Label One or Two
    ai_label = input("AI Label 1 or 2? (1/2): ").strip()
    if ai_label not in ('1', '2'):
        # TODO: Log bad input?
        exit(1)

    method = input("Perform skimming in addition to detection (and clipping for Label 1)? (y/n): ").strip().lower()
    sscd = SSCD(f"label{ai_label}")

    # Perform skimmer
    if method in ('y', 'yes'):
        search = input("Enter your search terms (separate by comma): ").strip()  # TODO: Log empty or bad input?
        rounds = int(input("Enter the number of rounds for sifting: "))
        folder = search.replace(' ', '-')
        search = search.split(',')
        folder = folder.split(',')
        sscd.skimmer(search, folder, rounds)

    sscd.load_dataset()
    # Perform segmentation; includes detection and clipping
    sscd.image_segmentation()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        stream=sys.stderr
    )

    main()
