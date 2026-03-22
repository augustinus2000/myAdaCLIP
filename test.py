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
        
        # === [디버그] results 구조 확인 ===
        print("\n=== results 구조 확인 ===")
        for cls_name, content in results.items():
            print(f"\nClass: {cls_name}")
            if isinstance(content, dict):
                print("Keys:", list(content.keys()))
                # 각 key의 첫 번째 샘플 타입만 출력
                for k, v in content.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        print(f"  {k}: type={type(v[0])}, len={len(v)}")
        
        print("\n=== metric_dict 구조 확인 ===")
        for cls_name, metrics in metric_dict.items():
            print(f"{cls_name}: {metrics.keys()}")

        # 샘플 하나만 출력
        sample_cls = next(iter(results))
        print("\n샘플 클래스:", sample_cls)
        if 'img_paths' in results[sample_cls]:
            print("  img_paths[0]:", results[sample_cls]['img_paths'][0])
        if 'anomaly_maps' in results[sample_cls]:
            amap = results[sample_cls]['anomaly_maps'][0]
            print("  anomaly_maps[0]: type:", type(amap))
        if 'gt_paths' in results[sample_cls]:
            print("  gt_paths[0]:", results[sample_cls]['gt_paths'][0])
        else:
            print("  gt_paths key 없음")

        # --- Top-K 기반 image-level score 계산 ---
        all_scores, all_labels, all_names = [], [], []

        for cls_name, content in results.items():
            if not isinstance(content, dict):
                continue

            amaps = content.get('anomaly_maps', [])
            labels = content.get('anomaly_labels', content.get('labels', []))  # key 이름 확인 필요
            cls_list = [cls_name] * len(amaps)

            for amap, lab, cname in zip(amaps, labels, cls_list):
                if hasattr(amap, "detach"):
                    amap = amap.detach().cpu().numpy()
                amap = np.array(amap)
                if amap.ndim == 3:  # (1,H,W) → (H,W)
                    amap = np.squeeze(amap, axis=0)

                flat = amap.reshape(-1)
                K = min(args.topk, flat.shape[0])
                score = np.mean(np.partition(flat, -K)[-K:])  # ✅ Top-K mean
                all_scores.append(score)
                all_labels.append(int(lab))
                all_names.append(cname)
        
        """
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
        """
        # --- 표 형식 출력 준비 ---
        headers = ["object", "auroc_px", "aupro_px", "ap_px", "f1_px", "iou",
                   "auroc_sp", "ap_sp", "f1_sp", "best_th"]
        table = []

        for cls_name, metrics in metric_dict.items():
            row = [
                cls_name,
                round(metrics['auroc_px'], 1),
                round(metrics['aupro_px'], 1),
                round(metrics['ap_px'], 1),
                round(metrics['f1_px'], 1),
                round(metrics['iou'], 1),
                round(metrics['auroc_im'], 1),
                round(metrics['ap_im'], 1),
                round(metrics['f1_im'], 1),
                round(metrics['best_th'], 3),
            ]
            table.append(row)

        print(tabulate(table, headers=headers, tablefmt="github"))

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
    parser.add_argument("--testing_data", type=str, default="mvtec", help="Dataset for testing (default: 'mvtec')")

    # for the image model
    parser.add_argument("--image_path", type=str, default="asset/img.png",
                        help="Model for testing (default: 'asset/img.png')")
    parser.add_argument("--class_name", type=str, default="bottle",
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

