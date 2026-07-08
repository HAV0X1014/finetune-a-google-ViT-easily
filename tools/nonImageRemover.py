import os
import shutil
from PIL import Image, UnidentifiedImageError
import sys

def find_and_move_non_images(source_dir, dest_folder_name="non_image_files", image_extensions=None):
    """
    Scans a directory, identifies non-image files (either by extension or by failing
    Pillow validation for files with image-like extensions), and moves them into
    a specified destination folder.

    Valid images (correct extension + successfully opened by Pillow) are left in place.
    All other files are moved to the non_image_files folder.
    """
    if image_extensions is None:
        # Common image extensions (lowercase, starting with dot)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico']

    print(f"--- Scanning Directory: {source_dir} ---")
    print(f"Image extensions for validation: {', '.join(image_extensions)}")
    print(f"Destination folder for non-images: {os.path.join(source_dir, dest_folder_name)}")
    print("-" * 25)

    # Validate source directory
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found or is not a directory: {source_dir}")
        return

    # Create destination directory if it doesn't exist
    destination_dir = os.path.join(source_dir, dest_folder_name)
    try:
        os.makedirs(destination_dir, exist_ok=True)
        print(f"Ensured destination directory exists: {destination_dir}")
    except OSError as e:
        print(f"Error creating destination directory {destination_dir}: {e}")
        return

    files_checked = 0
    files_with_image_ext = 0
    non_image_count = 0
    moved_count = 0
    errors_count = 0

    print("\nScanning files...")

    try:
        for entry in os.scandir(source_dir):
            files_checked += 1
            filepath = entry.path
            filename = entry.name

            # Skip directories and the destination folder itself
            if entry.is_dir():
                if filepath == destination_dir:
                    print(f"Skipping destination directory: {destination_dir}")
                else:
                    print(f"Skipping directory: {filepath}")
                continue

            # Get file extension (case-insensitive)
            _, file_extension = os.path.splitext(filename)
            ext_lower = file_extension.lower()

            is_image_candidate = ext_lower in image_extensions
            if is_image_candidate:
                files_with_image_ext += 1

            should_move = True

            if is_image_candidate:
                # For image-extension files, validate with Pillow
                try:
                    with Image.open(filepath) as img:
                        # Successfully opened → it's a valid image, keep it
                        should_move = False
                        # print(f"Valid image: {filename}")  # Uncomment for verbose output
                except UnidentifiedImageError:
                    print(f"NON-IMAGE (UnidentifiedImageError): {filename}")
                except Exception as e:
                    print(f"NON-IMAGE (Error opening: {e.__class__.__name__}): {filename}")
                    errors_count += 1
            else:
                # No image extension → treat as non-image
                print(f"NON-IMAGE (wrong extension): {filename}")

            if should_move:
                non_image_count += 1
                dest_filepath = os.path.join(destination_dir, filename)
                try:
                    if os.path.exists(dest_filepath):
                        print(f"Warning: Destination file exists, potentially overwriting: {filename}")
                    shutil.move(filepath, dest_filepath)
                    moved_count += 1
                    print(f"Moved -> {os.path.basename(destination_dir)}/{filename}")
                except OSError as move_error:
                    errors_count += 1
                    print(f"Error moving file {filename}: {move_error}")

    except Exception as scan_error:
        print(f"\nAn unexpected error occurred during directory scanning: {scan_error}")
        errors_count += 1

    print("\n--- Scan Complete ---")
    print(f"Total files and directories scanned: {files_checked}")
    print(f"Files matching image extensions: {files_with_image_ext}")
    print(f"Files identified as non-image: {non_image_count}")
    print(f"Files successfully moved: {moved_count}")
    print(f"Errors encountered (move or scan): {errors_count}")


# --- How to run the script ---

if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_directory_to_scan = sys.argv[1]
    else:
        source_directory_to_scan = input("Enter the path to the directory containing your files: ")

    destination_folder = "non_image_files"
    target_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico']

    find_and_move_non_images(source_directory_to_scan, destination_folder, target_extensions)
