from skimage.metrics import structural_similarity as ssim
import cv2

def compare_images(imageA, imageB):
    # Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM
    score, diff = ssim(grayA, grayB, full=True)
    return score

# Load the two input images
imageA = cv2.imread("khaki/Khaki0.jpg")
imageB = cv2.imread("khaki/Khaki1.jpg")

# Compare the images
similarity_score = compare_images(imageA, imageB)
print(f"Similarity Score: {similarity_score}")

# Determine if the images are very similar
threshold = 0.9  # You can adjust this threshold
if similarity_score > threshold:
    print("The images are very similar.")
else:
    print("The images are not very similar.")
