from PIL import Image
import numpy as np
from scipy.ndimage import label

# 대용량 이미지 제한 해제
Image.MAX_IMAGE_PIXELS = None

img = Image.open('input/stadium_test2.png').convert('RGBA')
arr = np.array(img)

# 알파 마스크 추출
alpha = arr[:, :, 3]
mask = (alpha > 10).astype(np.uint8)

# 8-connectivity 구조 커널
structure = np.ones((3, 3), dtype=np.int32)

# 연결 컴포넌트 레이블링
labeled, num_features = label(mask, structure=structure)

print('components', num_features)

# 각 컴포넌트 픽셀 수 + bbox 계산
components = []
for comp_id in range(1, num_features + 1):
    ys, xs = np.where(labeled == comp_id)
    count = len(ys)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    components.append({
        'bbox': bbox,
        'count': count,
    })

# 기존 출력 형식 그대로 유지
for i, comp in enumerate(sorted(components, key=lambda c: c['count'], reverse=True), 1):
    print(i, comp['count'], comp['bbox'])