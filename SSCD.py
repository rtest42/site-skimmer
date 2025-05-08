import logging
import json
import torch
import threading
import os
import sys
# import numpy as np
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
    def __init__(self, label: str) -> None:
        self.label = label
        self.categories = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress",
                           "Belt", "Left-shoe", "Right-shoe", "Scarf"]
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=device)
        self.dataset = None

    def load_dataset(self) -> None:
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
    def download_image(session: requests.Session, url: str, label: str, counter: list[int], lock: threading.Lock) -> None:
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

    def download_images(self, links: list[str], label: str) -> None:
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

    def skimmer(self, searches: list[str], folders: list[str], rounds: int) -> None:
        # Skimming is slow on purpose to prevent anti-bot detection
        for search, folder in zip(searches, folders):
            print(f"Searching {search} via Pinterest")
            result = subprocess.run(["node", "puppeteer-main.js", search, str(rounds)], capture_output=True, text=True)
            result = json.loads(result.stdout.strip())
            filtered_results = [url for url in result if '236x' in url]  # Only keep images with a width of 236px
            self.download_images(filtered_results, os.path.join(self.label, 'test', folder))

    def edge_detection(self, img: Image, threshold: float = 0.2) -> bool:
        pass

    def get_color_percentage(self, img: Image, clr: tuple = (255, 255, 255)) -> float:
        pass

    def image_segmentation(self) -> None:
        # Make directories
        for label_name in self.dataset.features['label'].names:
            for category in ["bad", "good", "output"]:
                os.makedirs(os.path.join(self.label, category, label_name), exist_ok=True)

        images = KeyDataset(self.dataset, "image")
        for i, output in tqdm(enumerate(self.pipe(images)), total=len(self.dataset), desc="Segmenting images"):
            data = self.dataset[i]
            image = data['image']
            filename = os.path.basename(data['filename']['path'])
            label_name = data['label-name']
            masks = {segment['label']: segment['mask'] for segment in output}

            if self.label == 'label1':
                upper_body = masks.get('Upper-clothes')
                lower_body = masks.get('Pants')
                if upper_body and lower_body:
                    pass
                elif upper_body and not lower_body:
                    pass
                elif not upper_body and lower_body:
                    pass
                else:
                    # TODO log image masks unsuccessful
                    continue
            elif self.label == 'label2':
                # Test if background surrounds main image completely
                background = masks.get('Background')
                if background:
                    # TODO check if inverse of background touches edge
                    # Inverse mask
                    background = background.point(lambda x: 255 - x)
                    # image.putalpha(background)

            image.save(os.path.join(self.label, 'output', label_name, filename))
            # Logic for handling masks
            # Detection
            #category = 'bad'
            #if 'Face' in labels and ('Left-shoe' in labels or 'Right-shoe' in labels):
            #    category = 'good'

            #image.save(os.path.join(category, label_name, filename))

            # Clipping (only for Label 1)
            #if self.label == 'label1':
            #    if len(masks) == 0:
            #        print("Skipping-no masks detected")  # TODO change to log
            #        continue

            #    stack = np.array(masks[0])
            #    for mask in masks:
            #        stack += np.array(mask)

            #    image.putalpha(Image.fromarray(stack))
            #    image.save(os.path.join(self.label, 'output', label_name, filename))


def main() -> int:
    ai_label = input("AI Label 1 or 2? (1/2): ").strip()
    if ai_label not in ('1', '2'):
        print("Input is not 1 or 2", file=sys.stderr)
        return 1

    sscd = SSCD(f"label{ai_label}")
    method = input("Perform skimming in addition to detection (and clipping for Label 1)? (y/n): ").strip().lower()
    # Perform skimmer
    if method in ('y', 'yes'):
        search = input("Enter your search terms (separate by comma): ").strip()
        if len(search) == 0:
            print("Search terms are empty", file=sys.stderr)
            return 1

        rounds = int(input("Enter the number of rounds for sifting: "))
        folder = search.replace(' ', '-')
        search = search.split(',')
        folder = folder.split(',')
        sscd.skimmer(search, folder, rounds)

    sscd.load_dataset()
    # Perform segmentation; includes detection and clipping
    sscd.image_segmentation()
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        stream=sys.stderr
    )

    code = main()
    print(f"Program finished with exit code {code}")
