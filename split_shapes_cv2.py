import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, math, zipfile

# PIL DecompressionBomb 경고 해제
Image.MAX_IMAGE_PIXELS = None

SRC = 'input/stadium_test2.png'
_base = 'output/shapes_split_cv2'
OUTDIR = _base
_counter = 1
while os.path.exists(OUTDIR):
    OUTDIR = f'{_base}_{_counter}'
    _counter += 1

os.makedirs(OUTDIR, exist_ok=True)

# ── 이미지 로드 & 연결 컴포넌트 ─────────────────────────────────
img = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)  # BGRA
if img is None:
    raise RuntimeError('failed to read image')

alpha = img[:, :, 3]
_, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

components = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area >= 1000:
        components.append((i, x, y, w, h, area))

components.sort(key=lambda t: (t[2], t[1]))
print(f'유효 컴포넌트 (area≥1000): {len(components)}')

manifest_lines = [f'meaningful_components={len(components)}']


# ── SVG path 변환 헬퍼 ──────────────────────────────────────────
def contour_to_svg_path(contour):
    """윤곽선 좌표를 SVG path d 속성으로 변환"""
    pts = contour.reshape(-1, 2)
    d = f'M {pts[0][0]} {pts[0][1]}'
    for pt in pts[1:]:
        d += f' L {pt[0]} {pt[1]}'
    d += ' Z'
    return d


# ── 컴포넌트별 SVG 저장 ──────────────────────────────────────────
for idx, (label_id, x, y, w, h, area) in enumerate(components, 1):
    crop = img[y:y+h, x:x+w].copy()
    local = (labels[y:y+h, x:x+w] == label_id)
    crop[~local] = [0, 0, 0, 0]

    # 윤곽선 추출
    alpha_ch = crop[:, :, 3].copy()
    _, binary = cv2.threshold(alpha_ch, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 대표 색상 추출 (BGRA → RGB)
    alpha_mask = crop[:, :, 3] > 127
    if alpha_mask.any():
        bgr = crop[:, :, :3]
        mean_color = bgr[alpha_mask].mean(axis=0).astype(int)
        fill_color = f'rgb({mean_color[2]},{mean_color[1]},{mean_color[0]})'  # BGR → RGB
    else:
        fill_color = 'black'

    # SVG path 생성
    svg_paths = []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            svg_paths.append(contour_to_svg_path(approx))

    name = f'shape_{idx:02d}.svg'
    out_path = os.path.join(OUTDIR, name)
    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n'
        f'  <path d="{" ".join(svg_paths)}" fill="{fill_color}" fill-rule="evenodd"/>\n'
        f'</svg>'
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    manifest_lines.append(f'{name}\tarea={area}\tbbox=({x},{y},{x+w-1},{y+h-1})')

# ── manifest 저장 ────────────────────────────────────────────────
with open(os.path.join(OUTDIR, 'manifest.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(manifest_lines))

# ── Contact sheet ────────────────────────────────────────────────
# cv2 BGRA → numpy RGBA (PIL 경유 없이 직접 변환)
img_rgba = img[:, :, [2, 1, 0, 3]]  # BGR→RGB 채널 재정렬, A 유지

thumb_paths_info = []
for idx, (label_id, x, y, w, h, area) in enumerate(components, 1):
    crop = img_rgba[y:y+h, x:x+w].copy()
    local = (labels[y:y+h, x:x+w] == label_id)
    crop[~local] = [0, 0, 0, 0]
    thumb_paths_info.append((crop, f'shape_{idx:02d}.svg', area))

cell_w, cell_h = 240, 280
cols = 4
rows = max(1, math.ceil(len(thumb_paths_info) / cols))
sheet = Image.new('RGB', (cell_w * cols, cell_h * rows), 'white')
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default()

for i, (crop_arr, svg_name, area) in enumerate(thumb_paths_info):
    r, c = divmod(i, cols)
    x0, y0 = c * cell_w, r * cell_h
    tile = Image.new('RGBA', (220, 220), (255, 255, 255, 255))
    im = Image.fromarray(crop_arr, 'RGBA')
    im.thumbnail((180, 180))
    px = (220 - im.width) // 2
    py = (180 - im.height) // 2 + 10
    tile.alpha_composite(im, (px, py))
    sheet.paste(tile.convert('RGB'), (x0 + 10, y0 + 10))
    draw.rectangle([x0 + 10, y0 + 10, x0 + 229, y0 + 229], outline=(200, 200, 200), width=1)
    draw.text((x0 + 10, y0 + 240), svg_name, fill='black', font=font)

sheet.save(os.path.join(OUTDIR, 'contact_sheet.png'))

# ── ZIP 압축 ─────────────────────────────────────────────────────
zip_path = f'{OUTDIR}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in sorted(os.listdir(OUTDIR)):
        full = os.path.join(OUTDIR, name)
        if os.path.isfile(full):
            zf.write(full, arcname=os.path.join('shapes_split', name))

print(f'meaningful_components={len(components)}')
for line in manifest_lines[1:]:
    print(line)
print(f'zip={zip_path}')