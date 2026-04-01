import os
import cv2  # 用于读取图片的真实宽和高，pip install opencv-python
from tqdm import tqdm

def main():
    #todo ===================== 1. 配置你的文件路径【无需改动，复制你的路径直接用】 =====================
    # visdrone
    # labels_dir = r"E:\daiyi\python\ultralytics\ultralytics\datasets\visdrone\VisDrone2019-DET-train\labels"
    # images_dir = r"E:\daiyi\python\ultralytics\ultralytics\datasets\visdrone\VisDrone2019-DET-train\images"
    # tinyperson
    # labels_dir = r"E:\daiyi\python\ultralytics\ultralytics\datasets\TinyPerson\labels\train"
    # images_dir = r"E:\daiyi\python\ultralytics\ultralytics\datasets\TinyPerson\images\train"
    # AITOD
    labels_dir = r"E:\daiyi\python\ultralytics\ultralytics\datasets\AI-TOD\labels\train"
    images_dir = r"E:\daiyi\python\ultralytics\ultralytics\datasets\AI-TOD\images\train"

    #todo ===================== 2. 定义目标面积判定阈值（通用标准，可根据需求自定义修改） =====================
    # 微小目标：面积 < 1024 像素² (32x32)
    # 小目标：1024 ≤ 面积 < 9216 像素² (32x32 ~ 96x96)
    # 中目标：9216 ≤ 面积 < 65536 像素² (96x96 ~ 256x256)
    # 大目标：面积 ≥ 65536 像素² (≥256x256)
    tiny_area = 16*16
    small_area = 32*32
    medium_area = 96*96
    tiny_num = 0
    small_num = 0
    medium_num = 0
    large_num = 0
    #todo ===================== 3. 遍历labels文件夹下所有的txt标注文件 =====================
    all_txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    for txt_filename in tqdm(all_txt_files, desc="正在遍历标注文件", unit="个文件"):
        # 只处理txt文件，过滤其他无关文件
        if not txt_filename.endswith(".txt"):
            continue

        # 拼接当前txt文件的完整路径
        txt_path = os.path.join(labels_dir, txt_filename)
        # 根据txt文件名，拼接对应图片的完整路径（自动匹配同名jpg/png）
        img_basename = os.path.splitext(txt_filename)[0]
        img_path_jpg = os.path.join(images_dir, f"{img_basename}.jpg")
        img_path_png = os.path.join(images_dir, f"{img_basename}.png")

        # 判断图片是jpg还是png，获取有效图片路径
        img_path = ""
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            print(f"⚠️  未找到[{txt_filename}]对应的图片文件，跳过")
            continue

        # ===================== 4. 获取图片的【真实宽度、真实高度】 =====================
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]  # opencv读取的是 高(height)、宽(width)

        # ===================== 5. 读取txt文件的每一行标注信息 =====================
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # print(f"\n=====================================")
        # print(f"📄 当前文件：{txt_filename} | 🖼️  对应图片：{img_basename} | 图片尺寸：{img_width}x{img_height}")
        # 遍历每一行标注数据
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()  # 去除行首尾的空格、换行符
            if not line:  # 跳过空行
                continue

            # 解析标注行：分割为 类别ID, x_center, y_center, w, h
            label_info = line.split()
            if len(label_info) != 5:
                print(f"❌ 第{line_num}行标注格式错误，跳过：{line}")
                continue

            # 数据类型转换：类别ID是整数，其余是浮点型(归一化值 0~1)
            cls_id, x_center, y_center, norm_w, norm_h = label_info
            cls_id = int(cls_id)
            x_center = float(x_center)
            y_center = float(y_center)
            norm_w = float(norm_w)
            norm_h = float(norm_h)

            # ===================== 6. 计算标注框的【真实宽、真实高、真实面积】 =====================
            box_real_w = img_width * norm_w  # 标注框真实宽度
            box_real_h = img_height * norm_h  # 标注框真实高度
            box_real_area = box_real_w * box_real_h  # 标注框真实面积

            # ===================== 7. Python的switch语法：match-case 判定目标大小 =====================

            match True:
                case _ if box_real_area <= tiny_area:
                    target_size = "【微小目标】"
                    tiny_num+=1
                case _ if tiny_area < box_real_area <= small_area:
                    target_size = "【小目标】"
                    small_num+=1
                case _ if small_area < box_real_area <= medium_area:
                    target_size = "【中目标】"
                    medium_num+=1
                case _:
                    target_size = "【大目标】"
                    large_num+=1

            # ===================== 8. 打印当前标注框的所有信息 =====================
            # print(
            #     f"第{line_num}行 | 类别ID：{cls_id} | 标注框真实尺寸：{box_real_w:.1f}x{box_real_h:.1f} | 真实面积：{box_real_area:.1f} 像素² | {target_size}")
    return  tiny_num, small_num, medium_num, large_num

if __name__=='__main__':
    tiny_num, small_num, medium_num, large_num =main()
    print(tiny_num, small_num, medium_num, large_num)
    #visdrone 89264 118341 116622 18978
    #tinyperson 19486 7483 3979 344
    #aitod 241245 35613 5722 0