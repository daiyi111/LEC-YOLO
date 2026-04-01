from ultralytics import YOLO,RTDETR
# todo 定型试验
# E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11s\weights\best.pt
model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11s\weights\best.pt')
# Run batched inference on a list of images
results = model([r'E:\daiyi\python\yolov13-main\yolov13-main\datasets\AI-TOD\test\images\P2331__1.0__1200___1159.png'])  # return a list of Results objects)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show(conf=False,labels=False,line_width=1)  # display to screen
    result.save(conf=False,labels=False,line_width=1,filename="result.jpg")  # save to disk