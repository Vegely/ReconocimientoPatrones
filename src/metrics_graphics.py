import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# ==============================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================
DATA_PATH = Path(r"C:\Users\Sergio\Desktop\TECI\Tecnicas Reconocimiento Patrones\testeo\yolo_raw_results_2.npz")
OUTPUT_DIR = Path(r"C:\Users\Sergio\Desktop\TECI\Tecnicas Reconocimiento Patrones\testeo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLDS = np.linspace(0, 1, 20)
IOU_THRESH = 0.5

# Diccionario de categorías completo
CATEGORY_MAPPING = {
    "persona": 0,
    "bicicleta": 1,
    "coche": 2,
    "moto": 3,
    "autobús": 4,
    "tren": 5,
    "camión": 6,
    "semáforo": 7,
    "boca de incendios": 8,
    "señal": 9,
    "perro": 10,
    "ciervo": 11,
    "monopatín": 12,
    "carrito": 13,
    "scooter": 14,
    "otro vehículo": 15
}

# Invertir diccionario
INDEX_TO_CATEGORY = {v: k for k, v in CATEGORY_MAPPING.items()}

# Clases específicas a analizar (solo persona y coche)
TARGET_CLASSES = ["persona", "coche"]
TARGET_INDICES = [CATEGORY_MAPPING[c] for c in TARGET_CLASSES]


# ==============================
# FUNCIONES DE MÉTRICAS
# ==============================
def compute_metrics(results, conf_thresh=0.25, iou_thresh=0.5):
    TP, FP, FN = 0, 0, 0
    iou_tp_list = []

    for r in results:
        pred_boxes = r["pred_boxes"]
        pred_classes = r["pred_classes"]
        pred_confs = r["pred_confs"]
        gt_boxes = r["gt_boxes"]
        gt_classes = r["gt_classes"]
        iou_matrix = r["iou_matrix"]

        keep = pred_confs >= conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        iou_matrix = iou_matrix[keep, :] if len(iou_matrix) > 0 else np.zeros((0, len(gt_boxes)))

        matched_gt = set()
        for i, pc in enumerate(pred_classes):
            if iou_matrix.shape[1] == 0:
                FP += 1
                continue

            valid_gt_indices = [j for j, gc in enumerate(gt_classes) if gc == pc and j not in matched_gt]

            if len(valid_gt_indices) == 0:
                FP += 1
                continue

            ious = iou_matrix[i, valid_gt_indices]
            jmax = np.argmax(ious)
            max_iou = ious[jmax]
            gt_match = valid_gt_indices[jmax]

            if max_iou >= iou_thresh:
                TP += 1
                matched_gt.add(gt_match)
                iou_tp_list.append(max_iou)
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(iou_tp_list) if iou_tp_list else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall,
        "F1": F1, "Mean_IoU_TP": mean_iou
    }


def compute_metrics_per_class(results, target_class=None, conf_thresh=0.5, iou_thresh=0.5):
    TP, FP, FN = 0, 0, 0
    iou_tp_list = []

    for r in results:
        pred_boxes = r["pred_boxes"]
        pred_classes = r["pred_classes"]
        pred_confs = r["pred_confs"]
        gt_boxes = r["gt_boxes"]
        gt_classes = r["gt_classes"]
        iou_matrix = r["iou_matrix"]

        keep = pred_confs >= conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        iou_matrix = iou_matrix[keep, :] if len(iou_matrix) > 0 else np.zeros((0, len(gt_boxes)))

        if target_class is not None:
            class_keep = pred_classes == target_class
            pred_boxes = pred_boxes[class_keep]
            pred_classes = pred_classes[class_keep]
            iou_matrix = iou_matrix[class_keep, :] if len(iou_matrix) > 0 else np.zeros((0, len(gt_boxes)))

        matched_gt = set()
        for i, pc in enumerate(pred_classes):
            if iou_matrix.shape[1] == 0:
                FP += 1
                continue
            ious = np.array([iou_matrix[i, j] for j in range(len(gt_boxes)) if target_class is None or gt_classes[j] == target_class])
            gt_indices = [j for j in range(len(gt_boxes)) if target_class is None or gt_classes[j] == target_class]
            if len(ious) == 0:
                FP += 1
                continue
            j = np.argmax(ious)
            max_iou = ious[j]
            if max_iou >= iou_thresh and gt_indices[j] not in matched_gt:
                TP += 1
                matched_gt.add(gt_indices[j])
                iou_tp_list.append(max_iou)
            else:
                FP += 1

        FN += sum((gt_classes[j] == target_class if target_class is not None else True) for j in range(len(gt_boxes)) if j not in matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(iou_tp_list) if iou_tp_list else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall,
        "F1": F1, "Mean_IoU_TP": mean_iou
    }


def calculate_ap(recall, precision):
    recall = recall[::-1]
    precision = precision[::-1]
    rec = np.array(recall)
    prec = np.array(precision)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


# ==============================
# EJECUCIÓN PRINCIPAL
# ==============================
def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    results = data["results"]

    # --------------------------
    # Métricas globales por clase y graficado
    # --------------------------
    classes = np.unique(np.concatenate([r["gt_classes"] for r in results]))
    ap_dict = {}

    for cls in classes:
        precisions, recalls, mean_ious, f1_scores = [], [], [], []
        for conf in CONF_THRESHOLDS:
            metrics = compute_metrics_per_class(results, target_class=cls, conf_thresh=conf, iou_thresh=IOU_THRESH)
            precisions.append(metrics["Precision"])
            recalls.append(metrics["Recall"])
            mean_ious.append(metrics["Mean_IoU_TP"])
            f1_scores.append(metrics["F1"])

        cat_name = INDEX_TO_CATEGORY.get(cls, f"Clase {cls}")
        ap = calculate_ap(recalls, precisions)
        ap_dict[cat_name] = ap

        plt.figure(figsize=(10,6))
        plt.plot(CONF_THRESHOLDS, precisions, marker='o', label='Precisión')
        plt.plot(CONF_THRESHOLDS, recalls, marker='s', label='Sensibilidad')
        plt.plot(CONF_THRESHOLDS, mean_ious, marker='^', label='IoU Medio TP')
        plt.plot(CONF_THRESHOLDS, f1_scores, marker='x', label='F1')
        plt.xlabel("Umbral de confianza")
        plt.ylabel("Valor")
        plt.title(f"Métricas vs Umbral de confianza - Categoría: {cat_name} (AP={ap:.3f})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"metrics_categoria_{cls}.png", dpi=300)
        plt.close()

    # --------------------------
    # mAP ponderada
    # --------------------------
    all_gt_classes = [gt for r in results for gt in r["gt_classes"]]
    class_counts = Counter(all_gt_classes)
    class_sizes = {INDEX_TO_CATEGORY[k]: v for k, v in class_counts.items()}
    for cat in ap_dict:
        class_sizes.setdefault(cat, 0)
    ap_values = np.array([ap_dict[k] for k in ap_dict])
    sizes = np.array([class_sizes[k] for k in ap_dict])
    weighted_map = np.sum(ap_values * sizes) / np.sum(sizes)
    print("mAP ponderada:", weighted_map)

    # --------------------------
    # Métricas globales vs umbral
    # --------------------------
    precisions, recalls, mean_ious, f1_scores = [], [], [], []
    for conf in CONF_THRESHOLDS:
        metrics = compute_metrics(results, conf_thresh=conf, iou_thresh=IOU_THRESH)
        precisions.append(metrics["Precision"])
        recalls.append(metrics["Recall"])
        mean_ious.append(metrics["Mean_IoU_TP"])
        f1_scores.append(metrics["F1"])

    plt.figure(figsize=(10,6))
    plt.plot(CONF_THRESHOLDS, precisions, marker='o', label='Precisión')
    plt.plot(CONF_THRESHOLDS, recalls, marker='s', label='Sensibilidad')
    plt.plot(CONF_THRESHOLDS, mean_ious, marker='^', label='IoU Medio TP')
    plt.plot(CONF_THRESHOLDS, f1_scores, marker='^', label='F1')
    plt.xlabel("Umbral de Confianza")
    plt.ylabel("Valor")
    plt.title(f"Métricas frente a Umbral de Confianza (mAP = {weighted_map:.3f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_vs_confidence.png", dpi=300)
    plt.close()

    # --------------------------
    # Curvas PR 
    # --------------------------

    plt.figure(figsize=(8, 6))
    
    for cat_name, cls_idx in CATEGORY_MAPPING.items():
        precisions, recalls = [], []
    
        for conf in CONF_THRESHOLDS:
            metrics = compute_metrics_per_class(
                results,
                target_class=cls_idx,
                conf_thresh=conf,
                iou_thresh=IOU_THRESH
            )
            p = metrics["Precision"]
            r = metrics["Recall"]
            precisions.append(p)
            recalls.append(r)
    
        precisions = np.array(precisions)
        recalls = np.array(recalls)
    
        # Quitar el punto (0,0)
        valid = ~((precisions == 0) & (recalls == 0))
        precisions = precisions[valid]
        recalls = recalls[valid]
    
        # Ordenar por recall
        order = np.argsort(recalls)
        recalls = recalls[order]
        precisions = precisions[order]
    
        # Calcular AP
        ap = calculate_ap(recalls, precisions)
    
        # Plot PR
        plt.plot(
            recalls,
            precisions,
            marker='o',
            linewidth=2,
            label=f"{cat_name}"
        )
    
    # Líneas de referencia en 0.5
    plt.axhline(0.5, linestyle='--', linewidth=1)
    
    plt.ylim(0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision–Recall para todas las categorías")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision-recall.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
