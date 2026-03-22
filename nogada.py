import cv2
import numpy as np

# 파일 경로
anomaly_map_path = '/home/junhyeok/document/adaclip/AdaCLIP/results/images/mvtec/zipper-1575_AdaCLIP.jpg'
gt_mask_path = '/home/junhyeok/document/adaclip/AdaCLIP/data/mvtec/zipper/ground_truth/broken_teeth/000_mask.png'
save_path = '/home/junhyeok/document/adaclip/AdaCLIP/nogada_results/zipper_overlay.jpg'

# 이미지 로드
anomaly_map = cv2.imread(anomaly_map_path)
gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)  # 흑백으로 로드

# GT 마스크를 anomaly map 크기로 리사이즈
gt_mask_resized = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)

# 외곽선 검출
contours, _ = cv2.findContours(gt_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 덮기
overlay = anomaly_map.copy()
cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)

# 저장
cv2.imwrite(save_path, overlay)
print(f"컨투어 포함 이미지 저장 완료: {save_path}")
