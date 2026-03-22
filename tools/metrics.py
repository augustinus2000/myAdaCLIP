import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, roc_auc_score
from scipy.ndimage import gaussian_filter
from skimage import measure

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def is_one_class(gt: np.ndarray):
    gt_ravel = gt.ravel()
    return gt_ravel.sum() == 0 or gt_ravel.sum() == gt_ravel.shape[0]


def calculate_px_metrics(gt_px, pr_px):
    if is_one_class(gt_px):  # In case there are only normal pixels or no pixel-level labels
        return 0, 0, 0

    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
    precisions, recalls, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

    return auroc_px * 100, f1_px * 100, ap_px * 100

"""
def calculate_im_metrics(gt_im, pr_im):
    if is_one_class(gt_im):  # In case there are only normal samples or no image-level labels
        return 0, 0, 0

    auroc_im = roc_auc_score(gt_im.ravel(), pr_im.ravel())
    precisions, recalls, _ = precision_recall_curve(gt_im.ravel(), pr_im.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_im = np.max(f1_scores[np.isfinite(f1_scores)])
    ap_im = average_precision_score(gt_im, pr_im)

    return ap_im * 100, auroc_im * 100, f1_im * 100
"""

def calculate_im_metrics(gt_px, pr_px, topk=2000):
    # 여기서 gt_im을 별도로 만들어야 함
    # 보통 image-level label은 gt_px의 OR-pooling으로 구할 수 있음
    gt_im = (gt_px.reshape(gt_px.shape[0], -1).sum(axis=1) > 0).astype(np.uint8)

    pr_im = []
    for amap in pr_px:
        flat = amap.flatten()
        k = min(topk, len(flat))
        topk_vals = np.partition(flat, -k)[-k:]
        pr_im.append(topk_vals.mean())
    pr_im = np.array(pr_im)

    if len(np.unique(gt_im)) == 1:
        return 0.0, 0.0, 0.0

    auroc_im = roc_auc_score(gt_im, pr_im)
    ap_im = average_precision_score(gt_im, pr_im)

    precisions, recalls, _ = precision_recall_curve(gt_im, pr_im)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    f1_im = f1_scores.max() if f1_scores.size else 0.0

    return ap_im * 100, auroc_im * 100, f1_im * 100



def calculate_average_metric(metrics: dict):
    average = {}
    for obj, metric in metrics.items():
        for k, v in metric.items():
            if k not in average:
                average[k] = []
            average[k].append(v)

    for k, v in average.items():
        average[k] = np.mean(v)

    return average

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step if max_step > 0 else 1.0
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[...] = amaps > th
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / (region.area + 1e-8))
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        denom = inverse_masks.sum()
        fpr = fp_pixels / denom if denom > 0 else 0.0
        pros.append(np.array(pro).mean() if len(pro) > 0 else 0.0)
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    pros = pros[idxes]
    if fprs.size == 0 or fprs.max() == fprs.min():
        return 0.0
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    return auc(fprs, pros)

def calculate_aupro_iou_threshold(gt_masks, anomaly_maps):
    gt = gt_masks.copy()
    pr = anomaly_maps.copy()
    if gt.ndim == 4 and gt.shape[1] == 1: gt = gt.squeeze(1)
    if pr.ndim == 4 and pr.shape[1] == 1: pr = pr.squeeze(1)
    gt = (gt > 0.5).astype(np.uint8)

    pr = gaussian_filter(pr, sigma=(0,8,8))  # FIX: no blur across batch
    pr_min = pr.min(axis=(1,2), keepdims=True)
    pr_max = pr.max(axis=(1,2), keepdims=True)
    pr = (pr - pr_min) / (pr_max - pr_min + 1e-8)

    aupro = cal_pro_score(gt, pr) * 100.0

    precisions, recalls, thresholds = precision_recall_curve(gt.ravel(), pr.ravel())
    f1 = (2*precisions*recalls)/(precisions+recalls+1e-8)
    best_th = thresholds[np.argmax(f1)] if thresholds.size > 0 else 0.5

    bin_pred = (pr >= best_th).astype(np.uint8)
    inter = np.logical_and(gt, bin_pred).sum()
    union = np.logical_or(gt, bin_pred).sum()
    iou = (inter / (union + 1e-8)) * 100.0
    return aupro, iou, best_th


# --- calculate_metric 내부 수정 (기존 코드 + 확장) ---
def calculate_metric(results, obj):
    gt_px, pr_px, gt_im, pr_im = [], [], [], []

    for idx in range(len(results['cls_names'])):
        if results['cls_names'][idx] == obj:
            gt_px.append(results['imgs_masks'][idx])
            pr_px.append(results['anomaly_maps'][idx])
            gt_im.append(results['imgs_gts'][idx])
            pr_im.append(results['anomaly_scores'][idx])

    gt_px = np.array(gt_px)
    pr_px = np.array(pr_px)
    gt_im = np.array(gt_im)
    pr_im = np.array(pr_im)

    # --- 픽셀 맵 전처리 (σ=8, per-image min–max) ---
    pr_px = gaussian_filter(pr_px, sigma=(0,8,8))
    pr_min = pr_px.min(axis=(1,2), keepdims=True)
    pr_max = pr_px.max(axis=(1,2), keepdims=True)
    pr_px = (pr_px - pr_min) / (pr_max - pr_min + 1e-8)

    # === 픽셀 레벨 지표 ===
    auroc_px, f1_px, ap_px = calculate_px_metrics(gt_px, pr_px)

    # === 이미지 레벨 지표 ===
    ap_im, auroc_im, f1_im = calculate_im_metrics(gt_im, pr_im)

    # === AUPRO / IoU / best_th ===
    aupro_px, iou_px, best_th = calculate_aupro_iou_threshold(gt_px, pr_px)

    metric = {
        'auroc_px': auroc_px,
        'ap_px': ap_px,
        'f1_px': f1_px,
        'aupro_px': aupro_px,
        'iou': iou_px,
        'best_th': best_th,
        'auroc_im': auroc_im,
        'ap_im': ap_im,
        'f1_im': f1_im,
    }

    return metric

"""
def calculate_metric(results, obj):
    gt_px = []
    pr_px = []

    gt_im = []
    pr_im = []

    for idx in range(len(results['cls_names'])):
        if results['cls_names'][idx] == obj:
            gt_px.append(results['imgs_masks'][idx])
            pr_px.append(results['anomaly_maps'][idx])

            gt_im.append(results['imgs_gts'][idx])
            pr_im.append(results['anomaly_scores'][idx])

    gt_px = np.array(gt_px)
    pr_px = np.array(pr_px)

    gt_im = np.array(gt_im)
    pr_im = np.array(pr_im)

    auroc_px, f1_px, ap_px = calculate_px_metrics(gt_px, pr_px)
    ap_im, auroc_im, f1_im = calculate_im_metrics(gt_im, pr_im)

    metric = {
        'auroc_px': auroc_px,
        'auroc_im': auroc_im,
        'f1_px': f1_px,
        'f1_im': f1_im,
        'ap_px': ap_px,
        'ap_im': ap_im,
    }

    return metric
"""
