from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import os, csv, cv2

MODEL_PATH = "./YOLO_weights/yolo11x/weights/best.pt"
TEST_IMAGES = "./test/img"
OUTPUT_CSV = "./YOLO_results/predictions.csv"
OUTPUT_IMG_DIR = "./YOLO_results/vis_images" 

CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
SKIP_BOX_THRESHOLD = 0.05

print("載入模型...")
model = YOLO(MODEL_PATH)

np.random.seed(42)
color_map = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)] 

def predict_with_tta(model, img_path, w, h, skip_box_thr):
    
    img = cv2.imread(img_path)
    all_boxes_norm = []
    all_scores = []
    all_labels = []

    res_orig = model.predict(source=img, conf=skip_box_thr, iou=0.99, verbose=False)[0]
    boxes_orig = res_orig.boxes.xyxy.cpu().numpy().copy()
    boxes_orig[:, [0, 2]] /= w
    boxes_orig[:, [1, 3]] /= h
    
    all_boxes_norm.append(boxes_orig)
    all_scores.append(res_orig.boxes.conf.cpu().numpy())
    all_labels.append(res_orig.boxes.cls.cpu().numpy())

    img_flipped = cv2.flip(img, 1)
    res_flipped = model.predict(source=img_flipped, conf=skip_box_thr, iou=0.99, verbose=False)[0]
    
    boxes_flipped = res_flipped.boxes.xyxy.cpu().numpy().copy()
    
    boxes_flipped[:, [0, 2]] /= w
    boxes_flipped[:, [1, 3]] /= h
    
    x1n_orig = 1.0 - boxes_flipped[:, 2] 
    x2n_orig = 1.0 - boxes_flipped[:, 0]
    
    boxes_flipped[:, 0] = x1n_orig
    boxes_flipped[:, 2] = x2n_orig
    
    all_boxes_norm.append(boxes_flipped)
    all_scores.append(res_flipped.boxes.conf.cpu().numpy())
    all_labels.append(res_flipped.boxes.cls.cpu().numpy())
    
    return all_boxes_norm, all_scores, all_labels

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

print("\n開始 TTA 預測並 WBF 融合...")

submission = []
image_files = sorted([os.path.join(TEST_IMAGES, f) for f in os.listdir(TEST_IMAGES) if f.endswith(('.jpg', '.png'))])


for image_path in image_files:
    
    img = cv2.imread(image_path)
    if img is None: continue
    h, w = img.shape[:2]
    image_id = int(os.path.splitext(os.path.basename(image_path))[0])
    
    boxes_list, scores_list, labels_list = predict_with_tta(model, image_path, w, h, SKIP_BOX_THRESHOLD)

    final_boxes, final_scores, final_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=IOU_THRESHOLD, skip_box_thr=SKIP_BOX_THRESHOLD
    )
    
    pred_strings = []
    
    for (x1n, y1n, x2n, y2n), score, cls in zip(final_boxes, final_scores, final_labels):
        if score < CONFIDENCE_THRESHOLD:
            continue

        x1_f, y1_f, x2_f, y2_f = x1n * w, y1n * h, x2n * w, y2n * h
        w_f, h_f = x2_f - x1_f, y2_f - y1_f

        x1_i, y1_i, x2_i, y2_i = map(int, [x1_f, y1_f, x2_f, y2_f])
        color = color_map[int(cls) % len(color_map)]

        cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        label = f"{int(cls)}:{score:.2f}"
        cv2.putText(img, label, (x1_i, max(y1_i - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        pred_strings.extend([
            f"{score:.6f}", f"{x1_f:.2f}", f"{y1_f:.2f}", f"{w_f:.2f}", f"{h_f:.2f}", str(int(cls))
        ])

    out_path = os.path.join(OUTPUT_IMG_DIR, f"{image_id}_11x_tta.jpg")
    cv2.imwrite(out_path, img)

    submission.append([image_id, " ".join(pred_strings)])

submission.sort(key=lambda x: x[0])
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_ID", "PredictionString"])
    writer.writerows(submission)

print(f"\n結果已儲存: {OUTPUT_CSV}")
print(f"圖像輸出位置: {OUTPUT_IMG_DIR}")