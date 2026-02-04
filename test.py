# import os
# import pandas as pd
# from FeatureExtraction import extract_squat_features_from_frames

# def process_folders_in_batches(root_path, view="side", batch_size=10, output_csv=None):
#     """
#     Process all frame folders in batches and save after each batch.

#     Args:
#         root_path (str): Folder containing subfolders for videos.
#         view (str): "front" or "side".
#         batch_size (int): Number of folders to process per batch.
#         output_csv (str): Path to CSV file to save results incrementally.

#     Returns:
#         List[dict]: Collected features for all folders.
#     """
#     all_features = []
#     folders = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))])
    
#     for i in range(0, len(folders), batch_size):
#         batch = folders[i:i+batch_size]
#         print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} folders) for {view} view...")
        
#         batch_features = []
#         for folder_name in batch:
#             folder_path = os.path.join(root_path, folder_name)
#             features = extract_squat_features_from_frames(folder_path, view=view)
#             batch_features.extend(features)
#             print(f"  Processed folder {folder_name}: {len(features)} reps")
        
#         all_features.extend(batch_features)

#         # Save CSV after each batch
#         if output_csv and batch_features:
#             df_batch = pd.DataFrame(all_features)
#             df_batch.to_csv(output_csv, index=False)
#             print(f"  Saved {len(all_features)} total {view} reps to {output_csv}")

#     return all_features


# # Dataset paths
# dataset_root = "dataset"
# front_root = os.path.join(dataset_root, "front")
# side_root = os.path.join(dataset_root, "side")

# batch_size = 10

# # Process front and side view folders
# front_features = process_folders_in_batches(front_root, view="front", batch_size=batch_size,
#                                             output_csv="front_view_features.csv")

# side_features = process_folders_in_batches(side_root, view="side", batch_size=batch_size,
#                                            output_csv="side_view_features.csv")

# print("\nProcessing complete!")
# print(f"Front view reps: {len(front_features)}")
# print(f"Side view reps: {len(side_features)}")
