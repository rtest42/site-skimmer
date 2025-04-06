import requests

def download_image(image_url, save_path):
    # Send a GET request to the image URL
    response = requests.get(image_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in write-binary mode and save the image
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded and saved to {save_path}")
    else:
        print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")

# Example usage:
image_url = input("Enter the image URL: ")
save_path = input("Enter the path to save the image (e.g., 'image.jpg'): ")

download_image(image_url, save_path)
