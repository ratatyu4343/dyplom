import imageio
from PIL import Image

# List of image file paths
image_paths = [f'./model_info/animation/prediction{i}.png' for i in range(100)]

# Load images and convert them to the same mode and size
images = []
for path in image_paths:
    img = Image.open(path).convert('RGBA')
    images.append(img)

# Save images as a GIF
output_path = 'output.gif'
imageio.mimwrite(f"{self}/output.gif", images, duration=1)