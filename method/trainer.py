import cv2
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from loss import FocalLoss, BinaryDiceLoss
from tools import visualization, calculate_metric, calculate_average_metric
from .adaclip import *
from .custom_clip import create_model_and_transforms
from skimage import measure
import os

def apply_ad_scoremap(img, scoremap, gt_mask=None, sigma=1, threshold=0.5):
    # gaussian smoothing
    scoremap = gaussian_filter(scoremap, sigma=sigma)

    # normalize scoremap
    scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min() + 1e-8)

    # === 클리핑으로 상위값 강조 ===
    clip_min = np.percentile(scoremap, 50)
    clip_max = np.percentile(scoremap, 99)
    scoremap = np.clip((scoremap - clip_min) / (clip_max - clip_min + 1e-8), 0, 1)

    # threshold mask
    mask = (scoremap > threshold).astype(np.uint8)
    mask_3ch = np.repeat(mask[:, :, None], 3, axis=2)

    # heatmap
    heatmap_uint8 = (scoremap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # overlay only where mask is 1
    overlay = np.where(mask_3ch == 1, heatmap, img)

    # blend
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # draw GT contours
    if gt_mask is not None:
        contours = measure.find_contours(gt_mask.astype(np.uint8), 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1).astype(np.int32)
            cv2.drawContours(blended, [contour], -1, (0, 255, 0), 2)

    return blended

class AdaCLIP_Trainer(nn.Module):
    def __init__(
            self,
            # clip-related
            backbone, feat_list, input_dim, output_dim,

            # learning-related
            learning_rate, device, image_size,

            # model settings
            prompting_depth=3, prompting_length=2,
            prompting_branch='VL', prompting_type='SD',
            use_hsf=True, k_clusters=20,
    ):

        super(AdaCLIP_Trainer, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        ########### different model choices
        freeze_clip, _, self.preprocess = create_model_and_transforms(backbone, image_size,
                                                                      pretrained='openai')
        freeze_clip  = freeze_clip.to(device)
        freeze_clip.eval()

        self.clip_model = AdaCLIP(freeze_clip=freeze_clip,
                                  text_channel=output_dim,
                                  visual_channel=input_dim,
                                  prompting_length=prompting_length,
                                  prompting_depth=prompting_depth,
                                  prompting_branch=prompting_branch,
                                  prompting_type=prompting_type,
                                  use_hsf=use_hsf,
                                  k_clusters=k_clusters,
                                  output_layers=feat_list,
                                  device=device,
                                  image_size=image_size).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size),
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          max_size=None)

        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        # update parameters
        self.learnable_paramter_list = [
            'text_prompter',
            'visual_prompter',
            'patch_token_layer',
            'cls_token_layer',
            'dynamic_visual_prompt_generator',
            'dynamic_text_prompt_generator'
        ]

        self.params_to_update = []
        for name, param in self.clip_model.named_parameters():
            # print(name)
            for update_name in self.learnable_paramter_list:
                if update_name in name:
                    # print(f'updated parameters--{name}: {update_name}')
                    self.params_to_update.append(param)

        # build the optimizer
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=learning_rate, betas=(0.5, 0.999))

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    # print(f'{param}: {update_name}')
                    self.save_dict[param] = value
                    break

        torch.save(self.save_dict, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(self, items):
        # print("[train_one_batch] 시작")

        image = items['img'].to(self.device)
        cls_name = items['cls_name']
        # print(f"[train_one_batch] image.shape: {image.shape}, cls_name: {cls_name}")

        # pixel level
        anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=False)
        # print("[train_one_batch] CLIP 모델 통과")

        if not isinstance(anomaly_map, list):
            anomaly_map = [anomaly_map]

        # losses
        gt = items['img_mask'].to(self.device)
        gt = gt.squeeze()

        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = items['anomaly'].to(self.device)
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0
        loss = 0

        # classification loss
        classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
        loss += classification_loss

        # seg loss
        seg_loss = 0
        for am, in zip(anomaly_map):
            seg_loss += (self.loss_focal(am, gt) + self.loss_dice(am[:, 1, :, :], gt) +
                         self.loss_dice(am[:, 0, :, :], 1-gt))

        loss += seg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_epoch(self, loader):
        self.clip_model.train()
        loss_list = []

        for idx, items in enumerate(tqdm(loader, desc="Training batches")):
            loss = self.train_one_batch(items)
            loss_list.append(loss.item())

        return np.mean(loss_list)

    
    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None):
        self.clip_model.eval()

        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_indx = 0
            for indx, items in enumerate(tqdm(dataloader, desc="Testing AdaCLIP", ncols=100)):
                if save_fig:
                    path = items['img_path']
                    for _path in path:
                        results['imgs'].append(_path)  # ✅ 경로만 저장
                    cls_name = items['cls_name']
                    for _cls_name in cls_name:
                        image_indx += 1
                        results['names'].append('{:}-{:03d}'.format(_cls_name, image_indx))

                image = items['img'].to(self.device)
                cls_name = items['cls_name']
                results['cls_names'].extend(cls_name)
                gt_mask = items['img_mask']
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                for _gt_mask in gt_mask:
                    results['imgs_masks'].append(_gt_mask.squeeze(0).numpy())  # px

                # pixel level
                anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=True)

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()

                for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                    # _anomaly_map = gaussian_filter(_anomaly_map, sigma=4)
                    results['anomaly_maps'].append(_anomaly_map)
                    results['anomaly_scores'].append(_anomaly_score)

                is_anomaly = np.array(items['anomaly'])
                for _is_anomaly in is_anomaly:
                    results['imgs_gts'].append(_is_anomaly)

        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = dict()
        
        for obj in tqdm(obj_list, desc='Calculating metrics per class', ncols=100):
            print(f"[INFO] Calculating metrics for class: {obj}")
            metric = calculate_metric(results, obj)
            obj_full_name = f'{obj}'
            metric_dict[obj_full_name] = metric

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        return metric_dict, results

