# Object-Detection-for-Group-hosed-Swine
# 114065448 通訊碩一 馬暐翔 HW1 Object Detection

## 安裝與準備（以下指令皆假設當前資料夾為 `HW1_114064558/`）

### 必要條件

環境：

  * **Python 3.11**
  * **CUDA 12.1**
  * **Torch 2.2.0+cu121**
  * **Torchvision 0.17.0+cu121**
  * **Numpy 1.26.4**
  * **ultralytics 8.3.206**
  * **ensemble-boxes 1.0.9**
  * **opencv-python 4.12.0.88**

**安裝依賴套件：**

```bash
pip install -r ./code_114064558/requirements.txt
```
-----
##  預期檔案結構

```bash
hw1_114064558
     |-------- report_114064558.pdf
     |-------- code_114064558  
          |-------- src
                    |-------- convert_to_yolo.py
                    |-------- yolo_v11x.py
                    |-------- yolov11x_test.py
          |-------- readme.md
          |-------- requirements.txt
     |-------- test
          |-------- img/
     |-------- train
          |-------- img/
          |-------- gt.txt
```
-----
##  執行流程（以下指令皆假設當前資料夾為 `HW1_114064558/`）
**若路徑出現錯誤請再換成絕對路徑執行**

### 1\. 資料格式轉換 (`convert_to_yolo.py`)

將原始資料集轉換為 Yolo 模型所需的標註格式。

```bash
python ./code_114064558/src/convert_to_yolo.py
```
> 轉換完後得切分好的 train 和 val 資料集以及 mydata.yaml

### 2\. 模型訓練 (`yolo_v11x.py`)

使用轉換後的資料集開始訓練 **YoloV11x** 模型。

```bash
python ./code_114064558/src/yolo_v11x.py
```

> 訓練完成後，最佳的模型權重會儲存至 `./YOLO_weights/yolo11x/weights/best.pt`。

### 3\. 模型測試與驗證 (`yolov11x_test.py`)

載入訓練好的權重，對測試集進行最終的效能評估和結果可視化。
> 預測完成，得 **predictions.csv** 和 **output_image**

```bash
python ./code_114064558/src/yolov11x_test.py
```

-----

## 預期輸出與結果

資料處理結果如：train, test 資料集 和 mydata.yaml 會儲存至 `./yolo_dataset`

訓練後最佳權重會儲存至 `./YOLO_weights/yolo11x/weights/best.pt`

預測結果如：**predictions.csv** ＆ **output_image** 會儲存至 `./YOLO_results/predictions.csv` 及 `./YOLO_results/vis_images`

| 輸出結果 | 說明 |
| :--- | :---
| **predictions.csv** | 符合kaggle競賽規定繳交格式的預測結果。 |
| **images** | 每張照片圈完Bounding Box的結果。 |
|
