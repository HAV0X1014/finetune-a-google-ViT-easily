import os

def delete_zero_byte_files(directory_path):
    """
    Scans the given directory for files with 0 bytes and offers to delete them.

    Args:
        directory_path (str): The path to the directory to scan.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found or is not a directory: {directory_path}")
        return

    print(f"Scanning directory: {directory_path}")

    zero_byte_files = []
    files_checked_count = 0
    errors_count = 0

    try:
        # List items in the directory
        items = os.listdir(directory_path)

        if not items:
            print("Directory is empty.")
            return

        for item_name in items:
            full_path = os.path.join(directory_path, item_name)

            # Check if it's a file and handle potential access errors
            try:
                if os.path.isfile(full_path):
                    files_checked_count += 1
                    # Get the size of the file
                    size = os.path.getsize(full_path)

                    # If size is 0, add to our list
                    if size == 0:
                        print(f"  Found 0-byte file: {item_name}")
                        zero_byte_files.append(full_path)
                    # Optional: Print for non-zero files (can be noisy)
                    # else:
                    #     print(f"  Checking file: {item_name} ({size} bytes)")

                # We ignore directories and other non-file items
            except OSError as e:
                # Handle errors during file info retrieval (e.g., permissions, broken symlinks)
                print(f"  Warning: Could not access or get info for {item_name}: {e}")
                errors_count += 1
            except Exception as e:
                 # Catch any other unexpected errors during file check
                 print(f"  Warning: Unexpected error checking {item_name}: {e}")
                 errors_count += 1


    except PermissionError:
        print(f"Error: Permission denied to access directory: {directory_path}")
        return
    except FileNotFoundError:
         # This case should ideally be caught by the initial isdir check
         print(f"Error: Directory not found during listing: {directory_path}")
         return
    except Exception as e:
        # Catch any other unexpected errors during directory listing
        print(f"An unexpected error occurred while listing directory contents: {e}")
        return


    print("-" * 30)
    print(f"Scan complete. {files_checked_count} files checked.")
    print(f"Found {len(zero_byte_files)} zero-byte files.")
    if errors_count > 0:
        print(f"Encountered {errors_count} errors accessing items.")
    print("-" * 30)


    if not zero_byte_files:
        print("No zero-byte files found in this directory.")
        return

    # --- Deletion Phase ---
    print("\nThe following zero-byte files were found:")
    for f in zero_byte_files:
        print(f"  - {f}")

    # Get confirmation before deleting
    confirmation = input("\nDo you want to delete these files? (yes/no): ").lower().strip()

    if confirmation == 'yes':
        deleted_count = 0
        print("\nAttempting to delete files...")
        for file_path in zero_byte_files:
            try:
                os.remove(file_path)
                print(f"  Successfully deleted: {os.path.basename(file_path)}")
                deleted_count += 1
            except OSError as e:
                print(f"  Error deleting {os.path.basename(file_path)}: {e}")
            except Exception as e:
                 print(f"  Unexpected error deleting {os.path.basename(file_path)}: {e}")


        print("-" * 30)
        print(f"Deletion complete. {deleted_count} files were deleted.")
        print(f"{len(zero_byte_files) - deleted_count} files could not be deleted.")
        print("-" * 30)

    else:
        print("Deletion cancelled by the user.")


# --- Main execution block ---
if __name__ == "__main__":
    # Prompt the user for the directory path
    target_directory = input("Enter the directory path to scan for 0-byte files: ")

    # Call the function to start the process
    delete_zero_byte_files(target_directory)
