import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class yolov11_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio, single_img=True):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
        model.load_state_dict(csd, strict=False)
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())
        self.total_saliency_map = None

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted_vals, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # 图片预处理
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 初始化特征和梯度提取器
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # 后处理预测结果
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
        valid_idx = torch.where(post_result.max(dim=1)[0] >= self.conf_threshold)[0]

        if len(valid_idx) == 0:
            print("⚠️ 警告：没有检测到符合置信度阈值的目标框！")
            return

        select_num = min(int(len(valid_idx) * self.ratio), len(valid_idx))
        valid_idx = valid_idx[:select_num]
        print(f"✅ 检测到 {len(valid_idx)} 个有效目标框，开始生成热力图...")

        # ========== 核心：删除clear_grads()，仅保留model.zero_grad()即可清空梯度 ==========
        for i in trange(len(valid_idx)):
            idx = valid_idx[i]
            self.model.zero_grad()  # 唯一梯度清空语句，所有版本通用，无报错！
            loss = 0.0

            # 计算反向传播的loss
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[idx].max()
                loss += score

            if self.backward_type == 'box' or self.backward_type == 'all':
                box_loss = pre_post_boxes[idx].sum()
                loss += box_loss

            # 反向传播获取梯度
            loss.backward(retain_graph=True)
            gradients = grads.gradients[0].cpu().detach().numpy()

            # ========== 适配所有版本的get_cam_weights完整传参，彻底无报错 ==========
            weights = self.method.get_cam_weights(self.method, tensor, None, activations, gradients)
            weights = weights.reshape((1, weights.shape[0], 1, 1))

            # 计算热力图
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))

            # 归一化热力图，防止除零错误
            if not np.isclose(saliency_map.max() - saliency_map.min(), 0):
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            else:
                continue

            # ========== 单张综合热力图融合（最优：最大值融合，适合小目标对比） ==========
            if self.single_img:
                if self.total_saliency_map is None:
                    self.total_saliency_map = saliency_map
                else:
                    self.total_saliency_map = np.maximum(self.total_saliency_map, saliency_map)
            else:
                cam_image = show_cam_on_image(img_float.copy(), saliency_map, use_rgb=True)
                cam_image = Image.fromarray(cam_image)
                cam_image.save(f'{save_path}/{i}.png')

        # 保存最终的单张综合热力图
        if self.single_img and self.total_saliency_map is not None:
            self.total_saliency_map = (self.total_saliency_map - self.total_saliency_map.min()) / (
                        self.total_saliency_map.max() - self.total_saliency_map.min())
            # image_weight=0.5 热力图更突出，方便对比C3k2改进效果
            cam_image = show_cam_on_image(img_float.copy(), self.total_saliency_map, use_rgb=True, image_weight=0.5)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/C3k2_heatmap_merged.png')
            print(f"✅ 成功生成【C3k2验证专用】单张热力图 → {save_path}/C3k2_heatmap_merged.png")
        elif not self.single_img:
            print(f"✅ 成功生成 {len(os.listdir(save_path))} 张热力图 → {save_path}")


def get_params():
    params = {
        'weight': r'E:\daiyi\python\ultralytics\ultralytics\tests\runs\detect\AI-TOD+yolo11s+RFAConv+C3k2_dff\weights\best.pt',
        'cfg': r'E:\daiyi\python\ultralytics\ultralytics\ultralytics\cfg\models\11\yolo11s-RFAConv-C3k2_DFF_1.yaml',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',  # ✅ 必选！对小目标的热力图细节最好，验证C3k2效果最明显
        'layer': 'model.model[8]',  # ✅ 你的C3k2改进核心层【骨干网第一个C3k2】，验证优先级最高
        'backward_type': 'all',  # ✅ 兼顾类别+框的注意力，最贴合检测任务
        'conf_threshold': 0.05,  # ✅ 适配AI-TOD小目标，低阈值能捕捉更多小目标
        'ratio': 0.3,  # ✅ 取30%高置信框，足够覆盖所有有效小目标
        'single_img': True  # ✅ 只输出1张融合图，完美解决多张问题，方便对比改进效果
    }
    return params


from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = yolov11_heatmap(**get_params())
    model(r'E:\daiyi\python\ultralytics\ultralytics\datasets\AI-TOD\test\images\P2331__1.0__1200___1159.png', 'result')