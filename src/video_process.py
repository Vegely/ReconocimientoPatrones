import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from collections import Counter


#### METRICAS ####

def compute_metrics(results, conf_thresh=0.25, iou_thresh=0.5):
    """
    Calcula métricas (TP, FP, FN, Precision, Recall, F1) 
    a partir de los resultados guardados de YOLO.
    
    Args:
        results: lista de diccionarios, cada uno con:
                 - pred_boxes
                 - pred_confs
                 - pred_classes
                 - gt_boxes
                 - gt_classes
                 - iou_matrix
        conf_thresh: float, umbral mínimo de confianza para considerar predicción
        iou_thresh: float, IoU mínimo para considerar TP
    
    Returns:
        dict con TP, FP, FN, Precision, Recall, F1
    """
    TP, FP, FN = 0, 0, 0
    iou_tp_list = []

    for r in results:
        pred_boxes = r["pred_boxes"]
        pred_classes = r["pred_classes"]
        pred_confs = r["pred_confs"]
        gt_boxes = r["gt_boxes"]
        gt_classes = r["gt_classes"]
        iou_matrix = r["iou_matrix"]

        # Filtrar por confidence
        keep = pred_confs >= conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        iou_matrix = iou_matrix[keep, :] if len(iou_matrix) > 0 else np.zeros((0, len(gt_boxes)))

        # Matching con chequeo de clase
        matched_gt = set()
        for i, pc in enumerate(pred_classes):
            if iou_matrix.shape[1] == 0:
                FP += 1
                continue
            
            # Solo considerar GT de la misma clase
            valid_gt_indices = [j for j, gc in enumerate(gt_classes) if gc == pc and j not in matched_gt]
            
            if len(valid_gt_indices) == 0:
                FP += 1
                continue
        
            # Tomar el GT con IoU máximo entre los válidos
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

        # FN = GT no matcheados
        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(iou_tp_list) if iou_tp_list else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": F1,
        "Mean_IoU_TP": mean_iou
    }


def calculate_ap(recall, precision):
    """
    Calculates Average Precision (AP) for a single class using 
    All-Points Interpolation (standard for COCO/VOC).
    
    Args:
        recall (list or np.array): List of recall values (0.0 to 1.0)
        precision (list or np.array): List of precision values (0.0 to 1.0)
        
    Returns:
        float: The AP value (0.0 to 1.0)
    """
    # 1. Convert to numpy arrays
    recall = recall[::-1]
    precision = precision[::-1]
    rec = np.array(recall)
    prec = np.array(precision)

    # 2. Add sentinel values to the start and end
    # mrec: standardizes the X-axis from 0.0 to 1.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    # mpre: standardizes the Y-axis. We pad with 0 at the start and end.
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # 3. Compute the Precision Envelope
    # Make precision monotonically decreasing.
    # We traverse backward and set the current precision to the max of 
    # itself and the next value. This fills "dips" in the graph.
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i + 1])

    # 4. Calculate Area Under Curve (AUC)
    # Find indices where recall changes (the "steps" in the graph)
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum of rectangular areas: (Width of Recall Step) * (Height of Precision)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap 



data = np.load(r"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\src\yolo_raw_results.npz", allow_pickle=True)
results = data["results"]

# # Ejemplo: metrics con IoU=0.5, conf=0.25
# metrics = compute_metrics(results, conf_thresh=0.5, iou_thresh=0.5)
# print(metrics)

# print(results["inference_time_sec"])

# # Extraer todos los tiempos
# inference_times = np.array([r["inference_time_sec"] for r in results])

# print("Tiempos por imagen (s):", inference_times)
# print("Tiempo medio por imagen (ms):", inference_times.mean() * 1000)
# print("FPS medio:", 1 / inference_times.mean())



def compute_metrics_per_class(results, target_class=None, conf_thresh=0.5, iou_thresh=0.5):
    """
    Calcula TP, FP, FN, Precision, Recall, F1, Mean IoU de los TP
    filtrando solo la clase `target_class`.
    
    Si target_class=None, calcula para todas las clases (como antes).
    """
    TP, FP, FN = 0, 0, 0
    iou_tp_list = []

    for r in results:
        pred_boxes = r["pred_boxes"]
        pred_classes = r["pred_classes"]
        pred_confs = r["pred_confs"]
        gt_boxes = r["gt_boxes"]
        gt_classes = r["gt_classes"]
        iou_matrix = r["iou_matrix"]

        # Filtrar por confidence
        keep = pred_confs >= conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        iou_matrix = iou_matrix[keep, :] if len(iou_matrix) > 0 else np.zeros((0, len(gt_boxes)))

        # Filtrar por clase
        if target_class is not None:
            class_keep = pred_classes == target_class
            pred_boxes = pred_boxes[class_keep]
            pred_classes = pred_classes[class_keep]
            iou_matrix = iou_matrix[class_keep, :] if len(iou_matrix) > 0 else np.zeros((0, len(gt_boxes)))

        # Matching
        matched_gt = set()
        for i, pc in enumerate(pred_classes):
            if iou_matrix.shape[1] == 0:
                FP += 1
                continue
            # Solo GT de la misma clase
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

        # FN = GT no matcheados
        FN += sum((gt_classes[j] == target_class if target_class is not None else True) for j in range(len(gt_boxes)) if j not in matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(iou_tp_list) if iou_tp_list else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": F1,
        "Mean_IoU_TP": mean_iou
    }


    
    
    
    
    
    
    
    
# métricas y graficos para distintas clases
    
    
conf_thresholds = np.linspace(0, 1, 20)
iou_thresh = 0.5

# Diccionario con nombres de categorías en español
category_mapping = {
    "persona":0,
    "bicicleta":1,
    "coche":2,
    "moto":3,
    "autobús":4,
    "tren":5,
    "camión":6,
    "semáforo":7,
    "boca de incendios":8,
    "señal":9,
    "perro":10,
    "ciervo":11,
    "monopatín":12,
    "carrito":13,
    "scooter":14,
    "otro vehículo":15
}

# Invertir diccionario para pasar de índice a nombre
index_to_category = {v:k for k,v in category_mapping.items()}

# Lista de clases presentes en GT
classes = np.unique(np.concatenate([r["gt_classes"] for r in results]))
ap_dict = {}
for cls in classes:
    precisions = []
    recalls = []
    mean_ious = []
    f1 = []

    for conf in conf_thresholds:
        metrics = compute_metrics_per_class(results, target_class=cls, conf_thresh=conf, iou_thresh=iou_thresh)
        precisions.append(metrics["Precision"])
        recalls.append(metrics["Recall"])
        mean_ious.append(metrics["Mean_IoU_TP"])
        f1.append(metrics["F1"])

    # Obtener nombre de categoría en español
    cat_name = index_to_category.get(cls, f"Clase {cls}")
    
    # Calcular AP
    ap = calculate_ap(recalls, precisions)
    ap_dict[cat_name] = ap

    # Plot en español
    plt.figure(figsize=(10,6))
    plt.plot(conf_thresholds, precisions, marker='o', label='Precisión')
    plt.plot(conf_thresholds, recalls, marker='s', label='Sensibilidad')
    plt.plot(conf_thresholds, mean_ious, marker='^', label='IoU Medio TP')
    plt.plot(conf_thresholds, f1, marker='x', label='F1')    
    plt.xlabel("Umbral de confianza")
    plt.ylabel("Valor")
    plt.title(f"Métricas vs Umbral de confianza - Categoría: {cat_name} (AP={ap:.3f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f"metrics_categoria_{cls}.png"
    plt.savefig(rf"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\src\graficas\metrics_categoria_{cls}.png", dpi=300)
    plt.show()
    



# mAP ponderada

all_gt_classes = []
for r in results:
    all_gt_classes.extend(r["gt_classes"])  # acumulamos todas las clases de GT

class_counts = Counter(all_gt_classes)  # {0: 1200, 1: 50, ...}
class_sizes = {index_to_category[k]: v for k, v in class_counts.items()}

# Asegurarnos que todas las categorías tienen tamaño
for cat in ap_dict:
    if cat not in class_sizes:
        class_sizes[cat] = 0

ap_values = np.array([ap_dict[k] for k in ap_dict])
sizes = np.array([class_sizes[k] for k in ap_dict])

weighted_map = np.sum(ap_values * sizes) / np.sum(sizes)
print("mAP ponderada:", weighted_map)





# metricas globales

conf_thresholds = np.linspace(0, 1, 20)  # 20 valores de 0 a 1
iou_thresh = 0.5

precisions = []
recalls = []
mean_ious = []
f1 = []

for conf in conf_thresholds:
    metrics = compute_metrics(results, conf_thresh=conf, iou_thresh=iou_thresh)
    precisions.append(metrics["Precision"])
    recalls.append(metrics["Recall"])
    mean_ious.append(metrics["Mean_IoU_TP"])
    f1.append(metrics["F1"])


# Plot
plt.figure(figsize=(10,6))
plt.plot(conf_thresholds, precisions, marker='o', label='Precisión')
plt.plot(conf_thresholds, recalls, marker='s', label='Sensibilidad')
plt.plot(conf_thresholds, mean_ious, marker='^', label='IoU Medio TP')
plt.plot(conf_thresholds, f1, marker='^', label='F1')
plt.xlabel("Umbral de Confianza")
plt.ylabel("Valor")
plt.title(f"Métricas frente a Umbral de Confianza (mAP = {weighted_map:.3f})")
plt.grid(True)
plt.legend()
plt.savefig(rf"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\src\graficas\metrics_vs_confidence.png", dpi=300)
plt.show()







# Clases que queremos analizar
target_classes = ["persona", "coche"]
class_indices = [category_mapping[c] for c in target_classes]

for cls_name, cls_idx in zip(target_classes, class_indices):
    y_true = []   # 1 si es GT positivo, 0 si no
    y_score = []  # confianza de la predicción
    
    for r in results:
        pred_boxes = r["pred_boxes"]
        pred_classes = r["pred_classes"]
        pred_confs = r["pred_confs"]
        gt_boxes = r["gt_boxes"]
        gt_classes = r["gt_classes"]
        iou_matrix = r["iou_matrix"]

        # Filtramos predicciones y GT de la clase
        pred_mask = pred_classes == cls_idx
        gt_mask = gt_classes == cls_idx

        pred_boxes_cls = pred_boxes[pred_mask]
        pred_confs_cls = pred_confs[pred_mask]
        iou_matrix_cls = iou_matrix[pred_mask, :][:, gt_mask] if len(pred_boxes_cls)>0 and np.sum(gt_mask)>0 else np.zeros((len(pred_boxes_cls), np.sum(gt_mask)))
        gt_boxes_cls = gt_boxes[gt_mask]

        # Para cada predicción, 1 si TP, 0 si FP
        matched_gt = set()
        for i, conf in enumerate(pred_confs_cls):
            if len(gt_boxes_cls)==0:
                # no hay GT de esta clase -> todo FP
                y_true.append(0)
                y_score.append(conf)
                continue
            
            # max IoU con GT
            ious = iou_matrix_cls[i]
            j = np.argmax(ious)
            max_iou = ious[j]
            if max_iou >= 0.5 and j not in matched_gt:
                y_true.append(1)  # TP
                y_score.append(conf)
                matched_gt.add(j)
            else:
                y_true.append(0)  # FP
                y_score.append(conf)
        
        # GT no detectados -> FN, se agregan con score 0
        for j, gb in enumerate(gt_boxes_cls):
            if j not in matched_gt:
                y_true.append(1)
                y_score.append(0)

    # ROC y AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curva ROC - Clase: {cls_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ROC_{cls_name}.png", dpi=300)
    plt.show()
    
    
    
    
    

# Categorías a analizar
category_mapping = {
    "persona": 0,
    "coche": 2
}

plt.figure(figsize=(8, 6))

all_recalls = []

for cat_name, cls in category_mapping.items():
    precisions = []
    recalls = []

    for conf in conf_thresholds:
        metrics = compute_metrics_per_class(
            results,
            target_class=cls,
            conf_thresh=conf,
            iou_thresh=iou_thresh
        )
        precisions.append(metrics["Precision"])
        recalls.append(metrics["Recall"])

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

    all_recalls.extend(recalls)

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
plt.axvline(0.5, linestyle='--', linewidth=1)

# Ajustes de ejes
plt.ylim(0, 1.05)   # eje Y fijo
plt.autoscale(enable=True, axis='x', tight=True)  # eje X ajustado al contenido

# Estética
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Curva Precision–Recall")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(rf"C:\Users\Sergio\Documents\GitHub\ReconocimientoPatrones\src\graficas\precision-recall.png", dpi=300)
plt.show()



# roc
# Categorías a analizar
category_mapping = {
    "persona": 0,
    "coche": 2
}

plt.figure(figsize=(8, 6))

all_x = []

for cat_name, cls in category_mapping.items():
    recalls = []
    precisions = []

    for conf in conf_thresholds:
        metrics = compute_metrics_per_class(
            results,
            target_class=cls,
            conf_thresh=conf,
            iou_thresh=iou_thresh
        )
        recalls.append(metrics["Recall"])
        precisions.append(metrics["Precision"])

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Quitar el punto (0,0)
    valid = ~((recalls == 0) & (precisions == 0))
    recalls = recalls[valid]
    precisions = precisions[valid]

    # ROC-like
    tpr = recalls
    fpr = 1 - precisions

    # Ordenar por FPR
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    all_x.extend(fpr)

    # AUC
    auc = np.trapz(tpr, fpr)

    # Plot ROC
    plt.plot(
        fpr,
        tpr,
        marker='o',
        linewidth=2,
        label=f"{cat_name} (AUC={auc:.3f})"
    )

# Línea aleatoria
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label="Aleatorio")

# Ajustes de ejes
plt.ylim(0, 1.05)
plt.autoscale(enable=True, axis='x', tight=True)

# Estética
plt.xlabel("False Positive Rate (1 − Precision)")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"Curva ROC-like coherente con PR (IoU={iou_thresh})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
    
