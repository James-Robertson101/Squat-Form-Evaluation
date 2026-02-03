import os

def rename_folders_sequentially(parent_folder):
    folders = [
        f for f in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, f)) and f.isdigit()
    ]

    # Sort numerically (important!)
    folders = sorted(folders, key=lambda x: int(x))

    for new_index, folder in enumerate(folders, start=1):
        old_path = os.path.join(parent_folder, folder)
        new_name = str(new_index)
        new_path = os.path.join(parent_folder, new_name)

        if old_path != new_path:
            os.rename(old_path, new_path)

    print(f"Renamed {len(folders)} folders successfully.")

rename_folders_sequentially(r"C:\Users\james\OneDrive\Documents\both datasets combined\All side View Squats")
