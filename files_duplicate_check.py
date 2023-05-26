import os

def find_duplicate_files(folder):
    # Dictionary to store filenames and their paths
    files_dict = {}

    # Recursive function to iterate through the subfolders
    def search_files(subfolder):
        for root, _, files in os.walk(subfolder):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file)

                if file_name in files_dict:
                    files_dict[file_name].append(file_path)
                else:
                    files_dict[file_name] = [file_path]

    # Start searching from the main folder
    search_files(folder)

    # Filter out duplicate files
    duplicate_files = {name: paths for name, paths in files_dict.items() if len(paths) > 1}

    return duplicate_files

# Provide the path to the main folder
main_folder = '/path/to/main/folder'
duplicate_files = find_duplicate_files(main_folder)

# Display the duplicate files
for name, paths in duplicate_files.items():
    print(f"Duplicate files ({name}):")
    for path in paths:
        print(path)
    print()
