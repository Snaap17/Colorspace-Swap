from image_processing import process_images

def main():
    source_image_path = './images/fish.png'
    target_image_path = './images/blackhole.png'
    data_channel = 0 # 0 for Hue, 1 for Saturation, 2 for Value

    process_images(source_image_path, target_image_path, data_channel)

if __name__ == "__main__":
    main()