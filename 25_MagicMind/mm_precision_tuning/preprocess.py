import numpy as np
import argparse
from PIL import Image
import sys, os
from torchvision import transforms
sys.path.append("../")

def get_one_image(img_file, bin_file):

    if ".JPEG" not in img_file:
        print("No Image files input ", img_file)
        return

    resize = 256
    crop = 224
    data_scale = 1.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])

    image = Image.open(img_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image)

    image_array = image.numpy().reshape((1,3,crop,crop))
    np.save(bin_file, image_array)
    return image_array.shape, image_array.dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_file", type=str, required=True, help="input image data")
    parser.add_argument(
        "--bin_file", type=str, default="/tmp/output/bin_input", help="path to save image binary file")
    FLAGS = parser.parse_args()

    shape, dtype = get_one_image(FLAGS.img_file,FLAGS.bin_file)
    print("shape:",shape)
    print("dtype:", dtype)

