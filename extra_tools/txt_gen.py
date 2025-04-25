import os

def create_image_list(parent_folder):
    # Extract the stem of the parent folder path
    stem_name = os.path.basename(os.path.normpath(parent_folder))
    bag_name = f"{stem_name}.bag"
    
    # Image folder path
    image_folder = os.path.join(parent_folder, "rgb")
    
    # Output file path in the parent folder
    output_file = os.path.join(parent_folder, "rgb.txt")
    
    valid_extensions = {".png", ".jpg", ".jpeg"}
    
    # Open the output file in write mode
    with open(output_file, "w") as file:
        # Write the header
        file.write("# color images\n")
        file.write(f"# file: '{bag_name}'\n")
        file.write("# timestamp filename\n")
        
        # List and sort all files in the rgb folder
        images = sorted(os.listdir(image_folder))
        
        # Loop through the sorted files and write to the text file
        for img_name in images:
            # Check if the file is a PNG image
            if any(img_name.lower().endswith(ext) for ext in valid_extensions):
                # Remove the file extension to get the timestamp
                timestamp = img_name.split(".")[0]
                # Write the formatted line to the file
                file.write(f"{timestamp} rgb/{img_name}\n")
    
    print(f"Image list saved at: {output_file}")

# Take the parent folder path as input
parent_folder = input("Enter the path to the parent folder containing the 'rgb' folder: \n")
create_image_list(parent_folder)
