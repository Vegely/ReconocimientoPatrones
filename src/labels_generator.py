import json
import os
import pandas as pd

def labelsGenerator(input_path, output_path):    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    categories = pd.DataFrame(data["categories"])
    categories = categories[["id", "name"]]
    
    images = pd.DataFrame(data["images"])
    images = images.drop(["extra_info"], axis=1)
    images["file_name"] = images["file_name"].str.replace(r"^data/", "", regex=True)
    images["file_name"] = images["file_name"].str.replace(r"\.[^.]+$", ".txt", regex=True)
    
    annotations = pd.DataFrame(data["annotations"])
    annotations[["bbox_x_min", "bbox_y_min", "bbox_width", "bbox_height"]] = pd.DataFrame(annotations["bbox"].tolist(), index=annotations.index)
    annotations = annotations.drop(["extra_info", "segmentation", "track_id", "iscrowd", "bbox"], axis=1, errors="ignore")
    
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
        if os.path.exists(file_path):
            continue
        yolo_lines = group.apply(
            lambda x: f"{int(x['category_id'])} {x['bbox_x_center_norm']:.6f} {x['bbox_y_center_norm']:.6f} {x['bbox_width_norm']:.6f} {x['bbox_height_norm']:.6f}",
            axis=1
        )
        
        with open(file_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
data_dir = "../data/"
subdirs =  [data_dir + d for d in ["images_thermal_train/", "images_thermal_val/", "video_thermal_test/"]]
input_paths = [d + "coco.json" for d in subdirs]
output_paths = ["labels_train/", "labels_val/", "labels_test/"]
output_paths = [data_dir + o for o in output_paths]
[os.makedirs(o, exist_ok=True) for o in output_paths]

for (input_path, output_path) in zip(input_paths, output_paths):
    labelsGenerator(input_path, output_path)
    
    
    