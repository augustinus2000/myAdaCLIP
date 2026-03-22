import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os
import torch
from scipy.ndimage import gaussian_filter
import cv2

from tabulate import tabulate

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# Importing from local modules
from tools import write2csv, setup_seed, Logger
from dataset import get_data, dataset_dict
from method import AdaCLIP_Trainer
from PIL import Image
import numpy as np

from tools.visualizer import visualizer

setup_seed(111)

def test(args):
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."

    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_fig = args.save_fig

    logger = Logger('log.txt')
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    config_path = os.path.join('./model_configs', f'{args.model}.json')
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = AdaCLIP_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters
    ).to(device)

    model.load(args.ckt_path)

    if args.testing_model == 'dataset':
        assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support {dataset_dict.keys()}"

        save_root = args.save_path
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        csv_path = os.path.join(csv_root, f'{args.testing_data}.csv')
        image_dir = os.path.join(image_root, f'{args.testing_data}')
        os.makedirs(image_dir, exist_ok=True)

        test_data_cls_names, test_data, _ = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False
        )

        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        metric_dict, results = model.evaluation(
            test_dataloader,
            test_data_cls_names,
            save_fig,
            image_dir,
        )

        
        # === 전체 이미지 시각화 ===
        print("\n=== Start Visualization ===")

        # 안전 체크 & 간단한 통계
        n_imgs = len(results.get('imgs', []))
        n_amaps = len(results.get('anomaly_maps', []))
        n_cls   = len(results.get('cls_names', []))
        print(f"[Viz] counts -> imgs:{n_imgs}, amaps:{n_amaps}, cls_names:{n_cls}")

        if n_imgs == 0:
            print("[Viz] results['imgs']가 비어 있습니다. evaluation() 리턴을 확인하세요.")
        else:
            for i in range(n_imgs):
                img_path   = results['imgs'][i]
                amap       = results['anomaly_maps'][i]
                cls_name   = results['cls_names'][i]
                specie_name= results['names'][i] if 'names' in results and i < len(results['names']) else None
                gt_path    = results['imgs_gts'][i] if 'imgs_gts' in results and i < len(results['imgs_gts']) else None

                # best_th 가져오기 (없으면 0.5)
                best_th = metric_dict[cls_name]['best_th'] if cls_name in metric_dict else 0.5

                # anomaly map을 (1, H, W) numpy로 변환
                if hasattr(amap, "detach"):
                    amap_np = amap.detach().cpu().numpy()
                else:
                    amap_np = np.array(amap)

                if amap_np.ndim == 2:
                    amap_np = amap_np[None, ...]
                elif amap_np.ndim > 3:
                    amap_np = np.squeeze(amap_np)
                    if amap_np.ndim == 2:
                        amap_np = amap_np[None, ...]

                # 시각화 호출 (AnomalyCLIP 스타일 시그니처)
                visualizer(
                    img_path,
                    anomaly_map=amap_np,
                    img_size=args.image_size,
                    save_path=args.save_path,
                    cls_name=cls_name,
                    gt_mask=results['imgs_masks'][i],   # ✅ 단일 mask
                    threshold=best_th,
                    index=i   # ✅ 추가
                )
        print("=== Visualization Done ===")

        # --- 표 헤더 (픽셀 % 스케일, 이미지 % 스케일 통일) ---
        headers = ["object", "auroc_px", "aupro_px", "ap_px", "f1_px", "iou", "auroc_sp", "ap_sp", "f1_sp", "best_th_px", "best_th_sp"]

        table = []
        best_th_sp_map = {}

        # 전역 배열로 한 번만 뽑아두기
        all_cls  = np.array(results['cls_names'])
        all_gts  = np.array(results['imgs_gts']).astype(int)
        all_maps = results['anomaly_maps']  # 리스트 (H,W) 또는 (1,H,W)

        # --- image-level Top-K 점수 미리 계산 (전체) ---
        topk_scores = []
        for amap in all_maps:
            a = amap.detach().cpu().numpy() if hasattr(amap,"detach") else np.asarray(amap)
            if a.ndim == 3 and a.shape[0] == 1: a = a[0]
            mn, mx = float(a.min()), float(a.max())
            a = (a - mn) / (mx - mn + 1e-8)
            flat = a.reshape(-1)
            K = min(args.topk, flat.shape[0])
            score = float(np.mean(np.partition(flat, -K)[-K:]))
            topk_scores.append(score)
        topk_scores = np.array(topk_scores)

        for cls_name, metrics in metric_dict.items():
            if cls_name == "Average": 
                continue  # 평균 행은 나중에 따로

            # --- 픽셀 지표/픽셀 임계값 (metric_dict에서 제공) ---
            auroc_px  = metrics['auroc_px']
            aupro_px  = metrics['aupro_px']
            ap_px     = metrics['ap_px']
            f1_px     = metrics['f1_px']
            iou       = metrics['iou']
            best_th_px= metrics['best_th']  # 픽셀용 best_th

            # --- 이 클래스에 속한 샘플 마스크 ---
            m = (all_cls == cls_name)

            # --- 이미지 레벨 (Top-K 점수 기반) ---
            pr_sp = topk_scores[m]
            gt_sp = all_gts[m]
            if pr_sp.size > 0 and len(np.unique(gt_sp)) > 1:
                precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
                f1_vals   = (2*precisions*recalls)/(precisions+recalls+1e-8)
                f1_sp     = float(np.nanmax(f1_vals)) if f1_vals.size else 0.0
                best_th_sp= float(thresholds[np.nanargmax(f1_vals)]) if thresholds.size else 0.5
                auroc_sp  = float(roc_auc_score(gt_sp, pr_sp))
                ap_sp     = float(average_precision_score(gt_sp, pr_sp))
            else:
                # 한 클래스에 라벨이 한쪽만 있으면 PR/F1/임계값 정의 불가
                f1_sp = 0.0; best_th_sp = 0.5; auroc_sp = 0.0; ap_sp = 0.0

            best_th_sp_map[cls_name] = best_th_sp

            table.append([
                cls_name,
                round(auroc_px,2), round(aupro_px,2), round(ap_px,2), round(f1_px,2), round(iou,2),
                round(auroc_sp*100,2), round(ap_sp*100,2), round(f1_sp*100,2),
                round(best_th_px,3),               
                round(best_th_sp,3),
            ])
        
        # --- 평균 행 추가 ---
        if "Average" in metric_dict:
            avg = metric_dict["Average"]
            # 픽셀 레벨 평균 (metric_dict가 이미 계산해둔 평균 사용)
            auroc_px  = avg['auroc_px']
            aupro_px  = avg['aupro_px']
            ap_px     = avg['ap_px']
            f1_px     = avg['f1_px']
            iou       = avg['iou']

            # 픽셀 임계값 평균: metric_dict에 없으면 per-class로 평균
            if 'best_th' in avg and isinstance(avg['best_th'], (int, float)):
                best_th_px_avg = float(avg['best_th'])
            else:
                best_th_px_list = [m['best_th'] for k, m in metric_dict.items()
                                   if k != "Average" and 'best_th' in m]
                best_th_px_avg = float(np.mean(best_th_px_list)) if len(best_th_px_list) else float('nan')

            # === 이미지 레벨 평균 (표에 들어간 값에서 계산: 이미 % 스케일임) ===
            auroc_sps, ap_sps, f1_sps = [], [], []
            best_th_sps = []

            for row in table:
                # row = [cls, auroc_px, aupro_px, ap_px, f1_px, iou,
                #        auroc_sp, ap_sp, f1_sp, best_th_px, best_th_sp]
                auroc_sps.append(row[6])   # auroc_sp (%)
                ap_sps.append(row[7])      # ap_sp (%)
                f1_sps.append(row[8])      # f1_sp (%)
                best_th_sps.append(row[10])  # best_th_sp (0~1)

            auroc_sp_avg = float(np.mean(auroc_sps)) if auroc_sps else 0.0
            ap_sp_avg    = float(np.mean(ap_sps)) if ap_sps else 0.0
            f1_sp_avg    = float(np.mean(f1_sps)) if f1_sps else 0.0
            best_th_sp_avg = float(np.mean(best_th_sps)) if best_th_sps else float('nan')

            table.append([
                "Average",
                round(auroc_px,1), round(aupro_px,1), round(ap_px,1), round(f1_px,1), round(iou,1),
                round(auroc_sp_avg,1), round(ap_sp_avg,1), round(f1_sp_avg,1),
                round(best_th_px_avg,3), round(best_th_sp_avg,3),
            ])

        print(tabulate(table, headers=headers, tablefmt="github"))
        
        # ========= per-object & dataset-level TP/TN =========
        global_preds, global_labels = [], []
        dataset_name = args.testing_data

        for obj in metric_dict.keys():
            if obj == "Average": 
                continue
            m = (all_cls == obj)
            scores = topk_scores[m]
            labels = all_gts[m]
            if scores.size == 0 or len(np.unique(labels)) < 2:
                continue

            thr = best_th_sp_map.get(obj, 0.5)
            preds = (scores >= thr).astype(int)

            tn = np.sum((preds==0)&(labels==0))
            tp = np.sum((preds==1)&(labels==1))
            fp = np.sum((preds==1)&(labels==0))
            fn = np.sum((preds==0)&(labels==1))

            total_normal   = tn + fp
            total_abnormal = tp + fn
            sensitivity = tp/(tp+fn+1e-8); specificity = tn/(tn+fp+1e-8)
            fp_rate = fp/(fp+tn+1e-8) if total_normal>0 else 0.0
            fn_rate = fn/(fn+tp+1e-8) if total_abnormal>0 else 0.0

            print("\n=== Image-level TP/TN Analysis ===")
            print(f"Dataset: {dataset_name}")
            print(f"Category: {obj}")
            print(f"Threshold: {thr:.3f}")
            print(f"Total normal images (GT=0): {total_normal}")
            print(f"Total abnormal images (GT=1): {total_abnormal}\n")
            print(f"True Positive Rate (Sensitivity): {sensitivity*100:.2f}%")
            print(f"True Negative Rate (Specificity): {specificity*100:.2f}%\n")
            print(f"False Positive Rate: {fp} / {total_normal} = {fp_rate*100:.2f}%")
            print(f"False Negative Rate: {fn} / {total_abnormal} = {fn_rate*100:.2f}%")

            global_preds.extend(preds.tolist())
            global_labels.extend(labels.tolist())

        if len(global_labels) > 0:
            global_preds  = np.array(global_preds)
            global_labels = np.array(global_labels)
            tn = np.sum((global_preds==0)&(global_labels==0))
            tp = np.sum((global_preds==1)&(global_labels==1))
            fp = np.sum((global_preds==1)&(global_labels==0))
            fn = np.sum((global_preds==0)&(global_labels==1))
            total_normal   = tn + fp
            total_abnormal = tp + fn
            sensitivity = tp/(tp+fn+1e-8); specificity = tn/(tn+fp+1e-8)
            fp_rate = fp/(fp+tn+1e-8) if total_normal>0 else 0.0
            fn_rate = fn/(fn+tp+1e-8) if total_abnormal>0 else 0.0

            print("\n=== Image-level TP/TN Analysis (ALL categories, using per-object thresholds) ===")
            print(f"Dataset: {dataset_name}")
            print(f"Total normal images (GT=0): {total_normal}")
            print(f"Total abnormal images (GT=1): {total_abnormal}\n")
            print(f"True Positive Rate (Sensitivity): {sensitivity*100:.2f}%")
            print(f"True Negative Rate (Specificity): {specificity*100:.2f}%\n")
            print(f"False Positive Rate: {fp} / {total_normal} = {fp_rate*100:.2f}%")
            print(f"False Negative Rate: {fn} / {total_abnormal} = {fn_rate*100:.2f}%")


    elif args.testing_model == 'image':
        assert os.path.isfile(args.image_path), f"Please verify the input image path: {args.image_path}"
        ori_image = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
        pil_img = Image.open(args.image_path).convert('RGB')

        img_input = model.preprocess(pil_img).unsqueeze(0).to(model.device)

        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(img_input, [args.class_name], aggregation=True)

        anomaly_map = anomaly_map[0, :, :]
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map.cpu().numpy()
        anomaly_score = anomaly_score.cpu().numpy()

        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)

        vis_map = cv2.hconcat([ori_image, vis_map])
        save_path = os.path.join(args.save_path, args.save_name)
        print(f"Anomaly detection results are saved in {save_path}, with an anomaly of {anomaly_score:.3f} ")
        cv2.imwrite(save_path, vis_map)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaCLIP", add_help=True)

    # Paths and configurations
    parser.add_argument("--ckt_path", type=str, default='/home/junhyeok/document/adaclip/AdaCLIP/workspaces/models/0s-pretrained-visa-ViT-L-14-336-SD-VL-D4-L5-HSF-K20_best.pth',
                        help="Path to the pre-trained model (default: weights/pretrained_mvtec_colondb.pth)")

    parser.add_argument("--testing_model", type=str, default="dataset", choices=["dataset", "image"],
                        help="Model for testing (default: 'dataset')")

    # for the dataset model
    parser.add_argument("--testing_data", type=str, default="mpdd", help="Dataset for testing (default: 'mvtec')")

    # for the image model
    parser.add_argument("--image_path", type=str, default="asset/img.png",
                        help="Model for testing (default: 'asset/img.png')")
    parser.add_argument("--class_name", type=str, default="bracket_black",
                        help="The class name of the testing image (default: 'bottle')")
    parser.add_argument("--save_name", type=str, default="test.png",
                        help="Model for testing (default: 'dataset')")


    parser.add_argument("--save_path", type=str, default='./results',
                        help="Directory to save results (default: './results')")

    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")

    parser.add_argument("--save_fig", type=str2bool, default=True,
                        help="Save figures for visualizations (default: True)")

    # Hyper-parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")

    # Prompting parameters
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'],
                        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')")
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'],
                        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')")

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation. If False, original class embedding is used (default: True)")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")

    parser.add_argument("--topk", type=int, default=2000, help="image-level score 계산 시 사용할 상위 픽셀 개수 (Top-K)")


    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    test(args)

