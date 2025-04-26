from PIL import Image
import os

input_folder = 'test'
output_folder = 'test2'
target_size = (224, 224)  # width, height

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = Image.open(input_path).convert('RGB')
            width, height = img.size

            # Calculate center crop
            aspect_target = target_size[0] / target_size[1]
            aspect_img = width / height

            if aspect_img > aspect_target:
                # Image is wider: crop sides
                new_width = int(height * aspect_target)
                left = (width - new_width) // 2
                img = img.crop((left, 0, left + new_width, height))
            else:
                # Image is taller: crop top/bottom
                new_height = int(width / aspect_target)
                top = (height - new_height) // 2
                img = img.crop((0, top, width, top + new_height))

            # Now resize to target
            img = img.resize(target_size, Image.LANCZOS)
            img.save(output_path)
            print(f"Cropped and resized: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
