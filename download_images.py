import threading
import requests
import os
import sys
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from requests.exceptions import InvalidURL, ReadTimeout


# Helper function to get all image HTML tags
def extract_tags_helper(file: str, tag='img') -> list:
    with open(file, 'r', encoding='utf8') as f:
        print(f"Opening {file}")
        data = f.read()
        parsed_data = BeautifulSoup(data, 'html.parser')
        tags = parsed_data.find_all(tag)
        return tags
    

# Extract tags from multiple files using multithreading
def extract_tags(files: list[str]) -> list[str]:
    tags = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_tags_helper, files)

    for result in results:
        tags.extend(result)
    
    return tags


# Helper function to download and save an image
def download_images_helper(session, url: str, label: str, counter: list[int], lock, headers: dict, threshold=2, timeout=10) -> None:
    # Check validity of image sources
    if not url:
        return
    
    url = url.partition('?')[0]

    if not url.startswith('http'):
        url = 'https:' + url

    extension = os.path.splitext(url)[-1].lower()
    if '.svg' in extension:
        return
    
    if '.' not in extension:
        extension = '.jpg'

    # Download the image
    try:
        data = session.get(url, headers=headers, timeout=timeout)
    except (InvalidURL, ReadTimeout) as e:
        print(e)
        return
    
    if data.status_code != 200:
        return
    
    # Check image requirements
    data = data.content
    byte = len(data)
    if byte < 8000:
        return
    
    try:
        with Image.open(BytesIO(data)) as img:
            img = img.convert('RGB')
            width, height = img.size
            if width / height > threshold:
                return
            
            grayscale = all(r == g == b for r, g, b in img.getdata())
            if grayscale:
                return
    except UnidentifiedImageError as e:
        print(e)
        return
    
    # Name file
    with lock:
        name = os.path.join(label, f"{label}{counter[0]}{extension}")
        print(f"Fetched image #{counter[0]}")
        counter[0] += 1

    # Write and save image
    with open(name, 'wb') as f:
        f.write(data)


# Download multiple images at once
def download_images(links: list[str], label: str) -> None:
    os.makedirs(label, exist_ok=True)
    session = requests.Session()
    counter = [0]
    lock = threading.Lock()
    header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0'
    }

    with ThreadPoolExecutor() as executor:
        for link in links:
            executor.submit(download_images_helper, session, link, label, counter, lock, header)


# For debugging
def main(args) -> None:
    if len(args) < 2:
        print("Usage: python3 download_images.py <pattern1> [pattern2] ...")
        return

    files = []
    patterns = args[1:]
    for pattern in patterns:
        files.extend(glob(pattern))

    if len(files) == 0:
        print("No files found!")
        return
    
    label = input("Enter a label: ")
    tags = extract_tags(files)
    download_images(tags, label)


if __name__ == '__main__':
    main(sys.argv)
