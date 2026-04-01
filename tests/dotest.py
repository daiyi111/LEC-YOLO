import torch.cuda

# from ultralytics.hotMap import yolov11_heatmap
from ultralytics import YOLO
import kagglehub
import os
from glob import glob

def main():

    # 加载模型
    # model = YOLO("D:\\pdyroject\\python\\ultralytics\\tests\\runs\\detect\\LDConv\\weights\\best.pt")
    # # 评估模型
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # print(metrics.box.map)  # map50-95
    # print(metrics.box.map50)  # map50
    # print(metrics.box.map75)  # map75
    # print(metrics.box.maps ) # a list contains map50-95 of each category

    # # 训练模型
    # model=YOLO('yolo11n.yaml')
    # model.train(data="coco128.yaml", epochs=50,imgsz=640,batch=8,workers=4)

    # 20250326 yolo11n.yaml
    # 加载模型
    # model = YOLO("D:\\pdyroject\\python\\ultralytics\\tests\\yolo11n.pt")

    # # # 评估模型
    # metrics = model.val()  # no arguments needed, dataset and settings remembered
    # model.predict("D:\\pdyroject\\python\\ultralytics\\datasets\\coco128\\images\\train2017\\000000000009.jpg","D:\\pdyroject\\python\\ultralytics\\datasets\\coco128\\images\\train2017")
    # model=torch.load("D:\\pdyroject\\python\\ultralytics\\tests\\runs\\detect\\yolo11n-80class-300epoch\\weights\\best.pt")

    # 20250329
    # 训练模型yolo11s.yaml
    # model=YOLO('yolo11s-AIFI.yaml')
    # 考虑设置optimiser
    # model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=8,workers=4)
    # 测试模型
    # model=YOLO('D:\\pdyroject\\python\\ultralytics\\tests\\runs\\detect\\train9\\weights\\best.pt')
    # model.val()

    # 20250401
    # model=YOLO('yolo11s-LDConv.yaml')
    # model.train(data="visdrone.yaml", epochs=200,imgsz=640,batch=4,workers=4)
    # 20250402
    # model=YOLO('yolo11s.yaml')
    # model.train(data="coco128.yaml", epochs=100,imgsz=640,batch=4,workers=4)
    # model=YOLO('D:\\pdyroject\\python\\ultralytics\\tests\\runs\\detect\\train27\\weights\\best.pt')
    # model.val()
    # model=YOLO(r'D:\pdyroject\python\ultralytics\tests\runs\detect\train12\weights\best.pt')
    # model=YOLO("yolo11s-DICAM.yaml")
    # model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=8,workers=4)
    # model.val()

#     20250416周三
#     20250421
#     C3K2-DFF-1模块改进
#     model=YOLO('yolo11s-C3k2_DFF_1.yaml')
#     model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=4,workers=2)
#     model=YOLO(r'D:\pdyroject\python\ultralytics\runs\detect\train2\weights\best.pt')
#     model.val()

#     20250421
#     APConv测试，替换卷积块，或者bottleneck块儿
#     感觉还得重新训练一次，免得出现误差
#     model=YOLO('yolo11s-C3k2_DFF_1-PConv.yaml')
#     model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=4,workers=2)
#     model=YOLO(r'D:\pdyroject\python\ultralytics\runs\detect\train6\weights\best.pt')
#     metrics = model.val()  # 无需参数，数据集和设置记忆
#     print(metrics.box.map)  # map50-95
#     print(metrics.box.map50)  # map50
#     print(metrics.box.map75)  # map75
#     print(metrics.box.maps)  # 包含每个类别的map50-95列表
#     20250422
#     损失函数改成Wasston损失那再测试一下是否有改进
#     model=YOLO('yolo11s-C3k2_DFF_1.yaml')
#     model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=4,workers=2)
#     model=YOLO(r'D:\pdyroject\python\ultralytics\runs\detect\train8\weights\best.pt')
#     metrics=model.val()
#     print(metrics.box.map)
#     20250423 引入了MSCAAttention 并且也有wasston损失函数
#     model=YOLO(r'D:\pdyroject\python\ultralytics\ultralytics\cfg\models\11\yolo11s-C3k2_DFF_1-MSCAAttention.yaml')
#     model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=4,workers=2)
#     model=YOLO(r'D:\pdyroject\python\ultralytics\runs\detect\train11\weights\best.pt')
#     model.val()
#     20250424 引入了改进的MSCAAttention 回到了CIOU损失函数
#     20250516使用PIOU进行yolo11s.yaml的改进，看看该损失函数是否有效
#     model = YOLO(r'E:\daiyi\python\ultralytics\ultralytics\ultralytics\cfg\models\11\yolo11s.yaml')
#     model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=16,workers=2)
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train6\weights\best.pt')
#     model.val()

#     20250522
#     修改新的损失函数shape-IOU  注意参数设置
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\ultralytics\cfg\models\11\yolo11s.yaml')
#     model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=16,workers=2)
#     20250525
#     测试集测试一下
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train9\weights\best.pt')
    # model.val()
    # 改进基于pixel的损失函数
    # model=YOLO('yolo11s.yaml')
    # model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=16,workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train17\weights\best.pt')
    # model.val()
    # 改进基于class和pixel的损失函数
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train26\weights\best.pt')
    # model=YOLO(r'yolo11s.yaml')
    # model.train(data="visdrone.yaml", epochs=300,imgsz=640,batch=16,workers=2)
    # 20250610
    # E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train31\weights\best.pt
    #测试
    # model = YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train31\weights\best.pt')
    # model.val()
    # model=YOLO('yolo11s.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s+mosaic+class_based_ciou\weights\best.pt')
    # model.val()
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s+class_based_ciou\weights\best.pt')
    # model.val()
    # model=YOLO("yolo11s.yaml")
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train2\weights\best.pt')

    #todo  ciou + yolo11s-C3K2_DFF_1
    # model=YOLO('yolo11s-C3K2_DFF_1.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train\weights\best.pt')
    # model.val()
    # todo 改进classBasedIou
    # model=YOLO('yolo11s.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-C2PSA_SENetV1+mosaic+ciou\weights\best.pt')
    # model.val()
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s+mosaic+c3k2_dff_1_version2+ciou_0.391(没什么效果)\weights\best.pt')
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train22\weights\best.pt')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model.val()

#     再次测试yolo11s-C3K2_DFF_1.yaml   已经感觉到前面四个c3k2没必要加dff了 效果不如c3k2
#     model=YOLO('yolo11s-MSFR.yaml')
#     model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
#todo 考虑再做一下实验，效果咋没提升     model=YOLO(r'yolo11s-C3k2_DFF_1_version3-PConv.yaml')
#     model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-MSFR+mosaic+ciou\weights\best.pt')
#     model.val()

#     todo EMA三种插入方式
#     model=YOLO("yolo11s-EMA_version1.yaml")
#     model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train\weights\best.pt')
#     model.val()
#     model=YOLO("yolo11s-EMA_version3.yaml")
#     model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s+mosaic+ciou_1_map50_0.393\weights\best.pt')
#     model.val()

    # todo 改进SPPF
    # model=YOLO('yolo11s-Improved_SPPF.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train\weights\best.pt')
    # model.val()
    # 上述实验效果不明显
    # todo 改进卷积 RFAConv 效果不错 0.399
    # model=YOLO('yolo11s-RFAConv.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-RFAConv+mosaic+ciou_0.399\weights\best.pt')
    # model.val()
    # todo 改进卷积和改进C3K2_DFF融合
    # model=YOLO('yolo11s-RFAConv-C3K2_DFF_1.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=16, workers=2)
    # model=YOLO(r"E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train\weights\best.pt")
    # model.val()

    # todo  加检测头
    # model=YOLO('yolo11s-4head.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-4head+mosaic+ciou_0.439\weights\best.pt')
    # model.val()

    # todo 加p2检测头 减去p5检测头
    # model=YOLO('yolo11s-tinyhead.yaml')
    # model.train(data="visdrone.yaml", epochs=300, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-4head+mosaic+ciou_0.435\weights\best.pt')
    # model.val()

    # todo 试一下EIOU
    # model=YOLO('yolo11s.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train6\weights\best.pt')
    # model.val()

    # todo 试一下ECIOU (有效涨点)
    # model=YOLO('yolo11s.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train7\weights\best.pt')
    # model.val()
    # tensorboard --logdir = E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train6

    #todo yolo11s-tinyhead + ECIOU
    # 后续用ECIOU实验了
    # model=YOLO('yolo11s-tinyhead.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model= YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train6\weights\best.pt')
    # model.val()

    #todo yolo11s-ASF 这个改进模型试试效果 多加了个C3k2
    # model=YOLO('yolo11s-ImprovedASF.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model= YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train9\weights\best.pt')
    # model.val()

    #todo yolo11s-ASF 这个改进模型试试效果  +C3k2 +改进TFE
    # model=YOLO('yolo11s-ImprovedASF-RFAConv.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model= YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train10\weights\best.pt')
    # model.val()

    #todo yolo11s-ASF 这个改进模型试试效果  +C3k2 +改进TFE
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model= YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train12\weights\best.pt')
    # model.val()
    #todo yolo11s-tinyhead + RFAConv +C3K2_DFF_1 + ECIOU
    #上述实验完毕

    # todo 目前开始做tinyperson数据集
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml",epochs=300,imgsz=640,batch=8,workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train19\weights\best.pt')


    #原始yolo11s在tinyperson上测试
    #todo
    # model=YOLO('yolo11s.yaml')
    # model.train(data="tinyperson.yaml",epochs=500,imgsz=640,batch=8,workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train6\weights\best.pt')
    # model.val()
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s+RFAConv+C3k2_dff\weights\best.pt')
    # model.val()
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s\weights\best.pt')
    # model.val()
    # model=YOLO('yolo11s-RFAConv.yaml')
    # model.train(data="tinyperson.yaml", epochs=400, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s+RFAConv\weights\best.pt')
    # model.val()

    # todo 训练的时候epochs是300 后续可以改成400增加一下精度 因为好像没完全收敛 model=YOLO('yolo11s-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml", epochs=400, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train4\weights\best.pt')
    # model.val()

    # model=YOLO('yolo11s-ImprovedASF.yaml')
    # model.train(data="tinyperson.yaml", epochs=400, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train4\weights\best.pt')
    # model.val()

    # model=YOLO('yolo11s-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml", epochs=400, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train4\weights\best.pt')
    # model.val()
    # yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml", epochs=400, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train5\weights\best.pt')
    # model.val()
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml", epochs=500, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train5\weights\best.pt')
    # model.val()

    # 加个损失函数了 先Eiou 再Eiou+ciou
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml", epochs=500, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train5\weights\best.pt')
    # model.val()

    # eiou+ciou
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="tinyperson.yaml", epochs=500, imgsz=640, batch=8, workers=2)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s+RFAConv+C3k2_dff+MFA-FPN+EIoU\weights\best.pt')
    # model.val()

    # AI-TOD数据集
    # model=YOLO('yolo11s.yaml')
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=8)
    # model = YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train2\weights\best.pt')
    # model.val()

    # +C3k2_DFF_1
    # model=YOLO('yolo11s-C3k2_DFF_1.yaml')
    # model.train(data="AI-TOD.yaml", epochs=700, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11s+C3k2_dff\weights\best.pt')
    # model.val()

    #RFAConv
    # model=YOLO('yolo11s-RFAConv.yaml')
    # model.train(data="AI-TOD.yaml", epochs=700, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11s+RFAConv\weights\best.pt')
    # model.val()

    #todo
    # C3k2+RFAConv
    # model=YOLO('yolo11s-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="AI-TOD.yaml", epochs=700, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s\weights\best.pt')
    # model.val()

    # +损失函数 cciou
    # model=YOLO('yolo11s.yaml')
    # model.train(data="AI-TOD.yaml", epochs=800, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train4\weights\best.pt')
    # model.val()

    # model=YOLO('yolo11s-ImprovedASF.yaml')
    # model.train(data="AI-TOD.yaml", epochs=700, imgsz=640, batch=4,workers=4)


    # 生成模型的预测结果json和模型验证集的json
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s+RFAConv+C3k2_dff+MFA-FPN+CCIoU\weights\best.pt')
    # model.val(data="tinyperson.yaml",save_json=True)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s\weights\best.pt')
    # model.val(data="tinyperson.yaml",save_json=True)


    # 测试一下yolo11n模型
    # model = YOLO('yolo11n.yaml')
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11n\weights\best.pt')
    # model.val()

    #todo 重新跑
    # C3k2_DFF_1+RFAConv
    # model = YOLO('yolo11s-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train3\weights\best.pt')
    # model.val()

    #TFE模块
    # model=YOLO("yolo11s-ImprovedASF.yaml")
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=4)
    # model = YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train2\weights\best.pt')
    # model.val()

    #所有模块
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="AI-TOD.yaml",epochs=700,imgsz=640,batch=4,workers=1)
    # model = YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train11\weights\best.pt')
    # res=model.val()
    # print(res)
    #  +EIoU
    # model=YOLO('yolo11s.yaml')
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train32\weights\best.pt')
    # model.val()

    #  +EIoU+CIoU
    # model=YOLO('yolo11s.yaml')
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=8)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train33\weights\best.pt')
    # model.val()
    #  +ECIoU
    # model=YOLO('yolo11s.yaml')
    # model.train(data="AI-TOD.yaml", epochs=300, imgsz=640, batch=4)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11s\weights\best.pt')
    # model.val()

    #+ECIoU+RFAConv+C3k2_DFF+MFA-FPN
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="AI-TOD.yaml",epochs=300,imgsz=640,batch=4)
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train28\weights\best.pt')
    # model.val()

    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-ImprovedASF-RFAConv+mosaic_ciou_0.447\weights\best.pt')
    # model.val(data="visdrone.yaml",save_json=True)

    # 最好的模型 all
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8)
    # 然后就是model.val()
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\train3\weights\best.pt')
    # model.val()
    #最后考虑加损失函数 all+eciou  先试试EIoU
    # model=YOLO('yolo11s-ImprovedASF-RFAConv-C3k2_DFF_1.yaml')
    # model.train(data="visdrone.yaml",epochs=300,imgsz=640,batch=8)


#     测试推理速度，参数量，GFLOPS
#     model=YOLO('yolo11n.yaml')
#     model.train(data="AI-TOD.yaml", epochs=500, imgsz=640, batch=16)
#     model=YOLO('yolo11s.yaml')
#     model.train(data="AI-TOD.yaml", epochs=600, imgsz=640, batch=16)
#     model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s+mosaic+ciou_2_map50_0.39\weights\best.pt')
#     model.val(save_json=True)

    # todo 预测脚本 用于定型试验
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-ImprovedASF+mosaic+RFAConv+C3k2_DFF_1_0.451\weights\best.pt')
    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s+mosaic+ciou_2_map50_0.39\weights\best.pt')
    # results = model(
    #     [r"E:\daiyi\python\ultralytics\ultralytics\datasets\visdrone\VisDrone2019-DET-test-dev\images\9999947_00000_d_0000012.jpg"])  # return a list of Results objects
    # # 9999947_00000_d_0000012.jpg
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show(conf=False, labels=False)  # display to screen
    #     result.save(filename=r"E:\daiyi\python\ultralytics\ultralytics\runs\result.jpg",conf=False, labels=False,line_width=1)  # save to disk

    # model=YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\tinyperson+yolo11s\weights\best.pt')
    # model.val(save_json=True)

    model = YOLO(r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\yolo11s-tinyhead+mosaic+ciou_0.439\weights\best.pt')
    model.val()


if __name__ == '__main__':
    main()