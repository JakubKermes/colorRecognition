from bing_image_downloader import downloader
import os
import shutil

def download_images(keyword, output_path, limit=30):
    downloader.download(keyword, limit=limit, output_dir=output_path, adult_filter_off=False, force_replace=False)


    # Move images from subdirectory to the desired location
    subdirectory = os.path.join(output_path, keyword)
    for file_name in os.listdir(subdirectory):
        src_path = os.path.join(subdirectory, file_name)
        dest_path = os.path.join(output_path, file_name)

        # Check if the file already exists in the destination path
        if os.path.exists(dest_path):
            # You can choose to skip or replace the existing file
            # Here, I'm replacing the existing file
            os.remove(dest_path)

        shutil.move(src_path, dest_path)

    # Remove the unnecessary subdirectory
    os.rmdir(subdirectory)

def fill_folders_with_images(root_path, categories, limit_per_category=1):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(root_path, split)

        for category in categories:
            category_path = os.path.join(split_path, category)

            # Create category directory if it doesn't exist
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            download_images(category, category_path, limit=limit_per_category)

if __name__ == "__main__":
    dataset_root = "dataset"  # Updated dataset root path with leading slash
    categories = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "pink", "gray", "black"]
    limit_per_category = 30  # You can adjust the limit as needed

    fill_folders_with_images(dataset_root, categories, limit_per_category)
