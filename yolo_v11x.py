from ultralytics import YOLO

model = YOLO("yolo11x.yaml")  

train_results = model.train(
    data="./yolo_dataset/mydata.yaml",
    epochs=500,
    imgsz=640,
    batch=16,
    device=0,
    project="./YOLO_weights",
    name="yolo11x",
    exist_ok=True,
)

metrics = model.val()
print(metrics)
