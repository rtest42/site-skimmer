import os
import sys
import torch
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from torchvision import models, transforms
from PIL import Image, ImageFile


# Extract feature vector of an image
def extract_features_helper(path: str, model: models.vgg.VGG) -> torch.Tensor:
    with Image.open(path) as img:
        # Preprocess image
        img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        # Extract feature vector
        feature = model(tensor)
        print(f"Extracting image vector from {path}")
    
    return feature.flatten()


# Extract feature vectors for all images in a directory
def extract_features(directory: str, model: models.vgg.VGG) -> list[torch.Tensor]:
    features = []
    files = glob(f"{directory}/*")
    for file in files:
        features.append(extract_features_helper(file, model))

    return features


# Lists duplicate images
def list_duplicates(directory: str, features: list[torch.Tensor], threshold=0.95) -> list[str]:
    duplicates = set()
    removed = []
    files = glob(f"{directory}/*")

    for i in range(len(files)):
        if i in duplicates:
            continue

        for j in range(i + 1, len(files)):
            if j in duplicates:
                continue

            # Compare similarity
            score = torch.nn.functional.cosine_similarity(features[i], features[j], dim=0)
            if score.item() > threshold:
                print(f"Duplicate detected: {files[i]} and {files[j]} with score {score}")
                duplicates.add(j)
                removed.append(files[j])

    print(f"Duplicate images: {len(duplicates)}")
    return removed


# Removes duplicate images
def remove_duplicates(files: list[str]) -> None:
    with ThreadPoolExecutor() as executor:
        executor.map(os.remove, files)


# For debugging
def main(args=sys.argv) -> None:
    if len(args) < 2:
        print("Usage: python3 remove_duplicates.py <directory>")
        return
     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.isfile("vgg16-397923af.pth"):
        state_dict = torch.load("vgg16-397923af.pth")
        model = models.vgg16().to(device)
        model.load_state_dict(state_dict)
    else:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
        torch.save(model, "vgg16-397923af.pth")
    # model.classifier = model.classifier[0]  # fc1 layer
    model.eval()

    directory = args[1]
    features = extract_features(directory, model)
    files = list_duplicates(directory, features)
    remove_duplicates(list(files))


if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    main()
