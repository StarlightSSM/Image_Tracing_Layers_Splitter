import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, math, zipfile
SRC = 'input/stadium_test.png'
_base = 'output/shapes_split_cv2'
OUTDIR = _base
_counter = 1
while os.path.exists(OUTDIR):
    OUTDIR = f'{_base}_{_counter}'
    _counter += 1

os.makedirs(OUTDIR, exist_ok=True)

img = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)  # BGRA
if img is None:
    raise RuntimeError('failed to read image')
alpha = img[:,:,3]
# binary mask from alpha; strong threshold to suppress AA specks
_, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

components = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area >= 1000:
        components.append((i, x, y, w, h, area))

components.sort(key=lambda t: (t[2], t[1]))
manifest_lines = [f'meaningful_components={len(components)}']

for idx, (label_id, x, y, w, h, area) in enumerate(components, 1):
    crop = img[y:y+h, x:x+w].copy()
    local = (labels[y:y+h, x:x+w] == label_id)
    crop[~local] = [0,0,0,0]
    out_path = os.path.join(OUTDIR, f'shape_{idx:02d}.png')

    # 외곽선 처리 (벡터화 방식)
    alpha_ch = crop[:, :, 3].copy()

    # 1. 이진화
    _, binary = cv2.threshold(alpha_ch, 127, 255, cv2.THRESH_BINARY)

    # 2. 윤곽선 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 빈 마스크에 부드럽게 다시 그리기
    smooth_mask = np.zeros_like(alpha_ch)
    for contour in contours:
        # epsilon 값이 클수록 더 부드러워짐 (0.002 ~ 0.01 조절)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.fillPoly(smooth_mask, [approx], 255)

    crop[:, :, 3] = smooth_mask
    cv2.imwrite(out_path, crop)
    manifest_lines.append(f'shape_{idx:02d}.png\tarea={area}\tbbox=({x},{y},{x+w-1},{y+h-1})')

with open(os.path.join(OUTDIR, 'manifest.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(manifest_lines))

# contact sheet
thumb_paths = [os.path.join(OUTDIR, f'shape_{i:02d}.png') for i in range(1, len(components)+1)]
cell_w, cell_h = 240, 280
cols = 4
rows = max(1, math.ceil(len(thumb_paths)/cols))
sheet = Image.new('RGB', (cell_w*cols, cell_h*rows), 'white')
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default()
for i, p in enumerate(thumb_paths):
    r = i // cols
    c = i % cols
    x0, y0 = c*cell_w, r*cell_h
    tile = Image.new('RGBA', (220,220), (255,255,255,255))
    im = Image.open(p).convert('RGBA')
    im.thumbnail((180,180))
    px = (220 - im.width)//2
    py = (180 - im.height)//2 + 10
    tile.alpha_composite(im, (px, py))
    rgb = tile.convert('RGB')
    sheet.paste(rgb, (x0+10, y0+10))
    draw.rectangle([x0+10, y0+10, x0+229, y0+229], outline=(200,200,200), width=1)
    draw.text((x0+10, y0+240), os.path.basename(p), fill='black', font=font)

sheet.save(os.path.join(OUTDIR, 'contact_sheet.png'))

zip_path = 'output/shapes_split_cv2.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in sorted(os.listdir(OUTDIR)):
        full = os.path.join(OUTDIR, name)
        if os.path.isfile(full):
            zf.write(full, arcname=os.path.join('shapes_split', name))

print(f'meaningful_components={len(components)}')
for line in manifest_lines[1:]:
    print(line)
print(f'zip={zip_path}')