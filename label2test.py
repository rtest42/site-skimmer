import os
import sys
from image_segmentation import extract_masks
from pinscrape import scraper, Pinterest
from transformers import pipeline


# keyword = "full body clothing model"
# output_folder = "label2test"
# number_of_workers = os.cpu_count() or 1
# images_to_download = 100
# proxies = {}


def main(args):
    keyword = input("Enter your search keywords: ")
    output_folder = input("Enter your image directory: ")
    images_to_download = input("Enter the number of images to download (default 100): ")
    if len(images_to_download.strip()) == 0:
        images_to_download = 100
    else:
        images_to_download = int(images_to_download)
    proxies = {}
    number_of_workers = os.cpu_count() or 1

    details = scraper.scrape(keyword, output_folder, proxies, number_of_workers, images_to_download)
    p = Pinterest(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15")
    images_url = p.search(keyword, images_to_download)
    print(images_url)
    p.download(url_list=images_url, number_of_workers=number_of_workers, output_folder=output_folder)
    pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0)
    extract_masks(output_folder, [], pipe)


if __name__ == '__main__':
    main(sys.argv)
