# import torch

# # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# # # Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
# img = "prueba.jpg"  # Example image
# # img = "imagen.jpeg"

# # Perform inference (handles batching, resizing, normalization automatically)
# results = model(img)

# # Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
# results.print()  # Print results to console
# results.show()  # Display results in a window
# results.save()  # Save results to runs/detect/exp

import json
import os
import pandas as pd

ruta_json = "./coco.json"
output_path = "./labels/"

with open(ruta_json, 'r') as f:
    data = json.load(f)

categories = pd.DataFrame(data["categories"])
categories = categories[["id", "name"]]

images = pd.DataFrame(data["images"])
images = images.drop(["extra_info"], axis=1)
images["file_name"] = images["file_name"].str.replace(r"^data/", "", regex=True)
images["file_name"] = images["file_name"].str.replace(r"\.[^.]+$", ".txt", regex=True)

annotations = pd.DataFrame(data["annotations"])
annotations[["bbox_x_min", "bbox_y_min", "bbox_width", "bbox_height"]] = pd.DataFrame(annotations["bbox"].tolist(), index=annotations.index)
annotations = annotations.drop(["extra_info", "segmentation", "track_id", "iscrowd", "bbox"], axis=1)

df_merge = pd.merge(annotations, 
                    images[["id", "width", "height", "file_name"]], 
                    left_on="image_id", right_on="id", how="left", suffixes=("", "_img"))

df_merge = df_merge.rename(columns={"width" : "width_img", "height" : "height_img"})

df_merge["bbox_x_center"] = df_merge["bbox_x_min"] + df_merge["bbox_width"] / 2
df_merge["bbox_y_center"] = df_merge["bbox_y_min"] + df_merge["bbox_height"] / 2
df_merge[["bbox_x_center_norm", "bbox_width_norm"]] = df_merge[["bbox_x_center", "bbox_width"]].div(df_merge["width_img"], axis=0)
df_merge[["bbox_y_center_norm", "bbox_height_norm"]] = df_merge[["bbox_y_center", "bbox_height"]].div(df_merge["height_img"], axis=0)

df_merge["file_path"] = output_path + df_merge["file_name"]
for file_path, group in df_merge.groupby("file_path"):
    yolo_lines = group.apply(
        lambda x: f"{int(x['category_id'])} {x['bbox_x_center_norm']:.6f} {x['bbox_y_center_norm']:.6f} {x['bbox_width_norm']:.6f} {x['bbox_height_norm']:.6f}",
        axis=1
    )
    
    with open(file_path, "w") as f:
        f.write("\n".join(yolo_lines))
