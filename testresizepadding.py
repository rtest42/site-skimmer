from PIL import Image, ImageOps
import os

input_folder = 'test'
output_folder = 'test2'
target_size = (224, 224)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = Image.open(input_path).convert('RGB')

            # Resize while maintaining aspect ratio
            img.thumbnail(target_size, Image.LANCZOS)

            # Create a new image with the target size and paste the resized image into it
            padded_img = Image.new("RGB", target_size, (255, 255, 255))  # transparent padding
            left = (target_size[0] - img.size[0]) // 2
            top = (target_size[1] - img.size[1]) // 2
            padded_img.paste(img, (left, top))

            padded_img.save(output_path)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
