import os
import shutil
import random
import argparse

def move_random_files(source_folder, destination_folder, num_files):
    """
    Moves a specified number of random files from a source folder to a destination folder.

    Args:
        source_folder (str): The path to the folder containing the files.
        destination_folder (str): The path to the folder where the files should be moved.
        num_files (int): The number of random files to move (default: 150).
    """

    try:
        # Check if the source folder exists
        if not os.path.isdir(source_folder):
            raise ValueError(f"Source folder '{source_folder}' does not exist.")

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Destination folder '{destination_folder}' created.")

        # Get a list of all files in the source folder
        files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

        # Check if there are enough files in the source folder
        if len(files) < num_files:
            print(f"Warning: Only {len(files)} files found in the source folder, moving all of them.")
            num_files = len(files)

        # Randomly select files to move
        files_to_move = random.sample(files, num_files)

        # Move the files
        for file in files_to_move:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved: {file} from {source_folder} to {destination_folder}")
            except Exception as e:
                print(f"Error moving {file}: {e}")

        print(f"Successfully moved {num_files} files from {source_folder} to {destination_folder}.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move a specified number of random files from source to destination folder.")
    parser.add_argument("source_folder", help="Path to the source folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    parser.add_argument("-n", "--num_files", type=int, default=150, help="Number of random files to move (default: 150)")

    args = parser.parse_args()

    move_random_files(args.source_folder, args.destination_folder, args.num_files)
