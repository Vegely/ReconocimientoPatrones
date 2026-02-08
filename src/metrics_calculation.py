import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
import pathlib

# ---------- CONFIG (rutas y parámetros) ----------
MODEL_PATH = r"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\Pruebas\epocas_500\best.pt"
TEST_IMAGES = r"C:\Users\Sergio\Desktop\TECI\Tecnicas Reconocimiento Patrones\testeo\prueba\images\video_thermal_test"
TEST_LABELS = r"C:\Users\Sergio\Desktop\TECI\Tecnicas Reconocimiento Patrones\testeo\prueba\labels\video_thermal_test"
OUTPUT_DIR = r"C:\Users\Sergio\Desktop\TECI\Tecnicas Reconocimiento Patrones\testeo"
IOU_THRESHOLD = 0.5
SAVE_FILE = "yolo_raw_results_2.npz"

# ---------- IOU ----------
def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

# ---------- LABELS ----------
def read_labels(txt_path, img_shape):
    h, w = img_shape[:2]
    boxes, classes = [], []

    if not txt_path.exists():
        return np.empty((0, 4)), np.empty((0,), dtype=int)

    with open(txt_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            classes.append(int(cls))

    return np.array(boxes), np.array(classes)

# ---------- MAIN ----------
def main():
    # Crear carpeta de salida si no existe
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file_path = output_path / SAVE_FILE

    # Compatibilidad Windows Path en torch hub
    pathlib.PosixPath = pathlib.WindowsPath

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cargar modelo
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=True
    ).to(device)
    model.eval()

    results = []
    image_paths = list(Path(TEST_IMAGES).glob("*.*"))

    for img_path in tqdm(image_paths, desc="Inferencia sin conf threshold"):
        img = cv2.imread(str(img_path))

        # Medir tiempo de inferencia
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        yolo_out = model(img)

        if device == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0

        preds = yolo_out.xyxy[0].cpu().numpy()  # x1,y1,x2,y2,conf,cls
        pred_boxes = preds[:, :4]
        pred_confs = preds[:, 4]
        pred_classes = preds[:, 5].astype(int)

        # Ground truth
        label_path = Path(TEST_LABELS) / img_path.with_suffix(".txt").name
        gt_boxes, gt_classes = read_labels(label_path, img.shape)

        # Matriz IoU
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pb in enumerate(pred_boxes):
            for j, gb in enumerate(gt_boxes):
                if pred_classes[i] == gt_classes[j]:
                    iou_matrix[i, j] = bbox_iou(pb, gb)

        results.append({
            "image": img_path.name,
            "image_shape": img.shape,
            "inference_time_sec": infer_time,
            "pred_boxes": pred_boxes,
            "pred_confs": pred_confs,
            "pred_classes": pred_classes,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
            "iou_matrix": iou_matrix
        })
    
    # Imprime los tiempos
    inference_times = np.array([r["inference_time_sec"] for r in results])
    print("Tiempos por imagen (s):", inference_times)
    print("Tiempo medio por imagen (ms):", inference_times.mean() * 1000)
    print("FPS medio:", 1 / inference_times.mean())
    
    # Guardar resultados en la carpeta de salida
    np.savez_compressed(save_file_path, results=results)
    print(f"\nResultados completos guardados en: {save_file_path}")

# Ejecutar
if __name__ == "__main__":
    main()
