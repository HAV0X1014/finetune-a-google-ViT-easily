import os
import shutil
from PIL import Image, UnidentifiedImageError # Import specific error for clarity
import sys

def find_and_move_non_images(source_dir, dest_folder_name="non_image_files", image_extensions=None):
    """
    Scans a directory for files with image extensions, attempts to open them
    as images using Pillow, and moves files that fail validation into a
    specified destination folder.

    Args:
        source_dir (str): The path to the directory to scan.
        dest_folder_name (str): The name of the subfolder to create for non-image files.
        image_extensions (list, optional): A list of file extensions (lowercase,
                                           starting with '.') to check.
                                           Defaults to common image types.
    """
    if image_extensions is None:
        # Common image extensions (make sure they are lowercase and start with dot)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico']

    print(f"--- Scanning Directory: {source_dir} ---")
    print(f"Target extensions: {', '.join(image_extensions)}")
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

    # Use os.scandir for potentially better performance with large directories
    # os.listdir() is also fine, but scandir provides DirEntry objects
    try:
        for entry in os.scandir(source_dir):
            files_checked += 1
            filepath = entry.path
            filename = entry.name

            # Skip directories and the destination folder itself
            if entry.is_dir():
                 # Optionally skip the destination folder if found during scan
                if filepath == destination_dir:
                    print(f"Skipping destination directory: {destination_dir}")
                else:
                    print(f"Skipping directory: {filepath}") # Optional: report other subdirs
                continue

            # Check if file has a target image extension (case-insensitive)
            # os.path.splitext splits into (root, ext)
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() in image_extensions:
                files_with_image_ext += 1
                # print(f"Checking file: {filename}...") # Uncomment for verbose checking

                try:
                    # Attempt to open the file as an image
                    # We only need to open and close; Pillow validates on open
                    with Image.open(filepath) as img:
                        # If successful, it's likely a valid image.
                        # We don't need to do anything with the image object.
                        # print(f"{filename} seems to be a valid image.") # Uncomment for verbose checking
                        pass # Do nothing, it's an image
                except UnidentifiedImageError:
                    # Pillow couldn't identify the file as an image
                    non_image_count += 1
                    print(f"Identified as NON-IMAGE (UnidentifiedImageError): {filename}")
                    dest_filepath = os.path.join(destination_dir, filename)
                    try:
                         # Handle potential duplicate names in destination (simple overwrite warning)
                        if os.path.exists(dest_filepath):
                             print(f"Warning: Destination file exists, potentially overwriting: {filename}")
                        shutil.move(filepath, dest_filepath)
                        moved_count += 1
                        print(f"Moved -> {os.path.basename(destination_dir)}/{filename}")
                    except OSError as move_error:
                        errors_count += 1
                        print(f"Error moving file {filename}: {move_error}")
                except Exception as e:
                    # Catch any other errors during opening (e.g., corrupted file header beyond format detection)
                    non_image_count += 1
                    errors_count += 1 # Count this as an error during processing
                    print(f"Identified as NON-IMAGE (Error opening: {e.__class__.__name__}): {filename}")
                    dest_filepath = os.path.join(destination_dir, filename)
                    try:
                         # Handle potential duplicate names in destination (simple overwrite warning)
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
    # Get source directory from user input or command line argument
    if len(sys.argv) > 1:
        source_directory_to_scan = sys.argv[1]
    else:
        source_directory_to_scan = input("Enter the path to the directory containing your files: ")

    # Define the destination folder name (relative to the source directory)
    destination_folder = "non_image_files"

    # Define the extensions you care about (add/remove as needed)
    # Remember: these are just the *initial filter* based on file name
    target_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico']

    # Run the function
    find_and_move_non_images(source_directory_to_scan, destination_folder, target_extensions)
