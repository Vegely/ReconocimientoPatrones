# import torch

# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
# model = torch.hub.load(repo_or_dir='yolov5', model='custom', path='yolov5s.pt', source='local')

# # Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
#img = "images_thermal_train/data/video-2Af3dwvs6YPfwSSf6-frame-006450-qMZz6qvyHLyXZvCGj.jpg"  # Example image
# img = "imagen.jpeg"

# Perform inference (handles batching, resizing, normalization automatically)
# print(model)
#results = model(img)

# dummy_input = torch.randn(1, 3, 640, 640)

# 3. Export to ONNX
# torch.onnx.export(
#     model, 
#     dummy_input, 
#     "yolov5s.onnx", 
#     opset_version=18,
#     input_names=['images'],
#     output_names=['output']
# )

# print("Model exported to 'yolov5s.onnx'. Upload this file to https://netron.app to view.")
# Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
# results.print()  # Print results to console
# results.show()  # Display results in a window
# results.save()  # Save results to runs/detect/exp

# import json
# import os
# import pandas as pd

# ruta_json = "./coco.json"
# output_path = "./labels/"

# with open(ruta_json, 'r') as f:
#     data = json.load(f)

# categories = pd.DataFrame(data["categories"])
# categories = categories[["id", "name"]]

# images = pd.DataFrame(data["images"])
# images = images.drop(["extra_info"], axis=1)
# images["file_name"] = images["file_name"].str.replace(r"^data/", "", regex=True)
# images["file_name"] = images["file_name"].str.replace(r"\.[^.]+$", ".txt", regex=True)

# annotations = pd.DataFrame(data["annotations"])
# annotations[["bbox_x_min", "bbox_y_min", "bbox_width", "bbox_height"]] = pd.DataFrame(annotations["bbox"].tolist(), index=annotations.index)
# annotations = annotations.drop(["extra_info", "segmentation", "track_id", "iscrowd", "bbox"], axis=1)

# df_merge = pd.merge(annotations, 
#                     images[["id", "width", "height", "file_name"]], 
#                     left_on="image_id", right_on="id", how="left", suffixes=("", "_img"))

# df_merge = df_merge.rename(columns={"width" : "width_img", "height" : "height_img"})

# df_merge["bbox_x_center"] = df_merge["bbox_x_min"] + df_merge["bbox_width"] / 2
# df_merge["bbox_y_center"] = df_merge["bbox_y_min"] + df_merge["bbox_height"] / 2
# df_merge[["bbox_x_center_norm", "bbox_width_norm"]] = df_merge[["bbox_x_center", "bbox_width"]].div(df_merge["width_img"], axis=0)
# df_merge[["bbox_y_center_norm", "bbox_height_norm"]] = df_merge[["bbox_y_center", "bbox_height"]].div(df_merge["height_img"], axis=0)

# df_merge["file_path"] = output_path + df_merge["file_name"]
# for file_path, group in df_merge.groupby("file_path"):
#     yolo_lines = group.apply(
#         lambda x: f"{int(x['category_id'])} {x['bbox_x_center_norm']:.6f} {x['bbox_y_center_norm']:.6f} {x['bbox_width_norm']:.6f} {x['bbox_height_norm']:.6f}",
#         axis=1
#     )
    
#     with open(file_path, "w") as f:
#         f.write("\n".join(yolo_lines))

import torch

# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
img = r"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\data\video_rgb_test\data\video-RMxN6a4CcCeLGu4tA-frame-000731-q2KuuqphGsCeEbf5c.jpg"  # Example image

# Perform inference (handles batching, resizing, normalization automatically)
results = model(img)

# Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
results.print()  # Print results to console
results.show()  # Display results in a window
results.save()  # Save results to runs/detect/exp


import cv2

# Imagen original (asegúrate que sea la misma usada en inferencia)
image = cv2.imread(img)

for det in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = det

    # Convertir a int
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    # Dibujar bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color=(0, 255, 0),  # verde
        thickness=2
    )

    # Texto (clase + confianza)
    label = f"{results.names[int(cls)]} {conf:.2f}"
    # cv2.putText(
    #     image,
    #     label,
    #     (x1, y1 - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6,
    #     (0, 255, 0),
    #     2
    # )

import random

colors = {}

for det in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = det
    cls = int(cls)

    if cls not in colors:
        colors[cls] = [random.randint(0,255) for _ in range(3)]

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls], 2)



# Mostrar imagen
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



######### agrupacion videos ########

import cv2
import os

input_dir = r"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\data\video_rgb_test\data"
output_dir = r"C:\Users\Sergio\Desktop\videos_prueba"

from collections import defaultdict

def agrupar_por_frame_split(input_dir):
    grupos = defaultdict(list)

    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith(".jpg") and "-frame-" in file:
            clave = file.split("-frame-")[0]
            grupos[clave].append(os.path.join(input_dir, file))

    return grupos


grupos = agrupar_por_frame_split(input_dir)

def procesar_video(video_id, image_paths, model, output_dir, fps=25):
    os.makedirs(output_dir, exist_ok=True)

    # Leer primera imagen para tamaño
    first_img = cv2.imread(image_paths[0])
    h, w = first_img.shape[:2]

    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_path in image_paths:
        image = cv2.imread(img_path)

        results = model(image)

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            label = f"{results.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        writer.write(image)

    writer.release()
    print(f"Vídeo generado: {output_path}")


for i, (clave, frames) in enumerate(grupos.items()):
    print(f"Procesando grupo {i}: {len(frames)} frames")
    procesar_video(f"video_{i:03d}", frames, model, output_dir)
    