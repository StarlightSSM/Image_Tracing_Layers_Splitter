from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.ndimage import label, find_objects
import os, math
import cv2

# ── 대용량 이미지 제한 해제 ──────────────────────────────────────
Image.MAX_IMAGE_PIXELS = None

SRC = 'input/stadium_test2.png'
_base = 'output/shapes_split'
OUTDIR = _base
_counter = 1
while os.path.exists(OUTDIR):
    OUTDIR = f'{_base}_{_counter}'
    _counter += 1
os.makedirs(OUTDIR, exist_ok=True)

# ── 이미지 로드 & 마스크 ─────────────────────────────────────────
img = Image.open(SRC).convert('RGBA')
arr = np.array(img)
alpha = arr[:, :, 3]
mask = (alpha >= 128).astype(np.uint8)

# ── scipy로 연결 컴포넌트 레이블링 (8-connectivity) ──────────────
structure = np.ones((3, 3), dtype=np.int32)
labeled, num_features = label(mask, structure=structure)
print(f'전체 컴포넌트: {num_features}')

# ── 컴포넌트 정보 수집 (면적 필터: 1000px 이상) ──────────────────
slices = find_objects(labeled)
comps = []
for comp_id, sl in enumerate(slices, 1):
    if sl is None:
        continue
    sy, sx = sl
    miny, maxy = sy.start, sy.stop - 1
    minx, maxx = sx.start, sx.stop - 1
    area = int(np.sum(labeled[sl] == comp_id))
    if area < 1000:
        continue
    comps.append({
        'id': comp_id,
        'bbox': (minx, miny, maxx, maxy),
        'area': area,
        'slice': sl,
    })

# 위→아래, 왼→오른쪽 정렬 (기존 로직 유지)
comps.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
print(f'유효 컴포넌트 (area≥1000): {len(comps)}')


# ── SVG path 변환 헬퍼 ──────────────────────────────────────────
def contour_to_svg_path(contour):
    """윤곽선 좌표를 SVG path d 속성으로 변환"""
    pts = contour.reshape(-1, 2)
    d = f'M {pts[0][0]} {pts[0][1]}'
    for pt in pts[1:]:
        d += f' L {pt[0]} {pt[1]}'
    d += ' Z'
    return d


# ── 컴포넌트별 SVG 저장 + 썸네일 생성 ───────────────────────────
manifest = []
thumbs = []
thumb_labels = []

for idx, c in enumerate(comps, 1):
    minx, miny, maxx, maxy = c['bbox']
    comp_id = c['id']

    # 크롭 + 마스크 적용
    crop = arr[miny:maxy+1, minx:maxx+1].copy()
    local_mask = (labeled[miny:maxy+1, minx:maxx+1] == comp_id)
    crop[~local_mask] = [0, 0, 0, 0]

    # 윤곽선 추출
    alpha_ch = crop[:, :, 3].copy()
    _, binary = cv2.threshold(alpha_ch, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    svg_w = maxx - minx + 1
    svg_h = maxy - miny + 1

    # 대표 색상 추출
    rgb = crop[:, :, :3]
    alpha_mask = crop[:, :, 3] > 127
    if alpha_mask.any():
        mean_color = rgb[alpha_mask].mean(axis=0).astype(int)
        fill_color = f'rgb({mean_color[0]},{mean_color[1]},{mean_color[2]})'
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
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">\n'
        f'  <path d="{" ".join(svg_paths)}" fill="{fill_color}" fill-rule="evenodd"/>\n'
        f'</svg>'
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    manifest.append((name, c['area'], c['bbox']))

    # 썸네일 생성
    im = Image.fromarray(crop, 'RGBA')
    bg = Image.new('RGBA', (220, 220), (255, 255, 255, 255))
    im.thumbnail((180, 180))
    px = (220 - im.width) // 2
    py = 10 + (170 - im.height) // 2
    bg.alpha_composite(im, (px, py))
    a = np.array(bg)
    a[0, :, :3] = 220; a[-1, :, :3] = 220
    a[:, 0, :3] = 220; a[:, -1, :3] = 220
    a[:, :, 3] = 255
    thumbs.append(Image.fromarray(a, 'RGBA').convert('RGB'))
    thumb_labels.append(f"shape_{idx:02d}.svg\narea={c['area']}")

# ── Contact sheet ────────────────────────────────────────────────
font = ImageFont.load_default()
cell_w, cell_h = 220, 260
cols = 4
rows = math.ceil(len(thumbs) / cols) if thumbs else 1
sheet = Image.new('RGB', (cell_w * cols, cell_h * rows), 'white')
d = ImageDraw.Draw(sheet)
for i, thumb in enumerate(thumbs):
    r, c_idx = divmod(i, cols)
    x0, y0 = c_idx * cell_w, r * cell_h
    sheet.paste(thumb, (x0, y0))
    d.text((x0 + 10, y0 + 225), thumb_labels[i], fill='black', font=font)

sheet.save(os.path.join(OUTDIR, 'contact_sheet.png'))

# ── manifest ─────────────────────────────────────────────────────
with open(os.path.join(OUTDIR, 'manifest.txt'), 'w', encoding='utf-8') as f:
    f.write(f'meaningful_components={len(manifest)}\n')
    for name, area, bbox in manifest:
        f.write(f'{name}\tarea={area}\tbbox={bbox}\n')

print(f'saved {len(manifest)} components to {OUTDIR}')
for row in manifest:
    print(row)
