import os

def rename_images(folder_path):
    # List all files in the given folder
    files = sorted(os.listdir(folder_path))
    for file_name in files:
        # Check if the file is a PNG image
        if file_name.endswith(".png"):
            # Split the filename to extract the numeric part after the dash
            num_part = file_name.split("-")[1].split(".")[0]  # Get the number after "-"
            new_name = f"{int(num_part):010}.png"  # Zero-pad the number to 10 digits
            
            # Construct the new file path
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)
            
            # Rename the file
            try:
                os.rename(old_path, new_path)
                print(f"Renamed '{file_name}' to '{new_name}'")
            except Exception as e:
                print(f"Error renaming '{file_name}': {e}")
