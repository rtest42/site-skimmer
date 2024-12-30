import sys
from image_segmentation import extract_masks
from transformers import pipeline

def main(args):
    pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=0)
    extract_masks("imageslabel3", [], pipe)

if __name__ == '__main__':
    main(sys.argv)