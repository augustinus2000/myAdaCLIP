# tools/visualizer.py
import cv2, os, numpy as np

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    else:
        return (pred - min_value) / (max_value - min_value + 1e-8)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualizer(img_path, anomaly_map, img_size, save_path,
               cls_name, gt_mask=None, threshold: float = 0.5, index=None):
    """
    Args:
        img_path: str, 원본 이미지 경로
        anomaly_map: np.array or torch.Tensor, (H,W)
        cls_name: class name (str)
        gt_mask: np.array mask (H,W) or None
        threshold: float, anomaly threshold
    """
    # 원래 파일명
    filename = os.path.basename(img_path)
    base, ext = os.path.splitext(filename)

    # === 저장 경로 ===
    save_dir = os.path.join(save_path, "imgs", cls_name)
    os.makedirs(save_dir, exist_ok=True)

    # 파일명 고유화
    if index is not None:
        out_filename = f"{cls_name}_{index:04d}{ext}"
    else:
        out_filename = f"{cls_name}_{base}{ext}"

    out_path = os.path.join(save_dir, out_filename)


    # 원본 이미지 로드
    vis = cv2.cvtColor(
        cv2.resize(cv2.imread(img_path), (img_size, img_size)),
        cv2.COLOR_BGR2RGB
    )

    # anomaly map normalize + threshold
    m = np.array(anomaly_map.cpu().numpy() if hasattr(anomaly_map, "cpu") else anomaly_map)
    if m.ndim == 3:   # (1,H,W) -> (H,W)
        m = m.squeeze(0)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    binary = (m > threshold).astype(np.uint8)
    m = m * binary

    # heatmap overlay
    vis = apply_ad_scoremap(vis, m)

    # === GT contour (녹색) ===
    if gt_mask is not None:
        gt = cv2.resize(gt_mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY)  # 0/1이면 그대로
        contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)


    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[Visualizer] Saved: {out_path}")
