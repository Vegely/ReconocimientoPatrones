import cv2
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
OUTPUT_DIR = r"C:\Users\Sergio\Desktop\TECI\Tecnicas Reconocimiento Patrones\testeo"
TEST_IMAGES = r"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\data\images\video_thermal_test"
SAVE_FILE = r"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\src\yolo_raw_results.npz"
VIDEO_OUTPUT = "video_predicciones_vs_gt_punteado.mp4"
FPS = 25
SEGMENT_LENGTH = 3  # longitud de segmentos para el punteado

# ---------- Función para dibujar rectángulo punteado ----------
def draw_dashed_rectangle(img, top_left, bottom_right, color, thickness=1, segment_length=3):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Horizontal superior
    for x in range(x1, x2, segment_length*2):
        cv2.line(img, (x, y1), (min(x+segment_length, x2), y1), color, thickness)
    # Horizontal inferior
    for x in range(x1, x2, segment_length*2):
        cv2.line(img, (x, y2), (min(x+segment_length, x2), y2), color, thickness)
    # Vertical izquierda
    for y in range(y1, y2, segment_length*2):
        cv2.line(img, (x1, y), (x1, min(y+segment_length, y2)), color, thickness)
    # Vertical derecha
    for y in range(y1, y2, segment_length*2):
        cv2.line(img, (x2, y), (x2, min(y+segment_length, y2)), color, thickness)

# ---------- Cargar resultados ----------
data = np.load(Path(SAVE_FILE), allow_pickle=True)
results = data["results"]

# ---------- Preparar video ----------
image_paths = [Path(TEST_IMAGES) / r["image"] for r in results]
first_img = cv2.imread(str(image_paths[0]))
h, w = first_img.shape[:2]

video_writer = cv2.VideoWriter(
    str(Path(OUTPUT_DIR) / VIDEO_OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (w, h)
)

# ---------- Dibujar predicciones y GT ----------
for r in results:
    img_path = Path(TEST_IMAGES) / r["image"]
    image = cv2.imread(str(img_path))
    
    # Predicciones en verde sólido
    for box, cls, conf in zip(r["pred_boxes"], r["pred_classes"], r["pred_confs"]):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # GT en rojo punteado
    for box, cls in zip(r["gt_boxes"], r["gt_classes"]):
        x1, y1, x2, y2 = map(int, box)
        # label = f"GT {cls}"
        draw_dashed_rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1, segment_length=SEGMENT_LENGTH)
        # cv2.putText(image, label, (x1, max(20, y1 - 5)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # Leyenda fija en cada frame
    # cv2.putText(image, "Cajas del modelo", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    # cv2.putText(image, "Etiquetas reales", (10, 60),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    video_writer.write(image)

video_writer.release()
print(f"Video generado: {Path(OUTPUT_DIR) / VIDEO_OUTPUT}")
