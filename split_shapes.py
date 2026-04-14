from PIL import Image
import numpy as np
from collections import deque
import os, math
import cv2
SRC = 'input/stadium_test1.png'
_base = 'output/shapes_split'
OUTDIR = _base
_counter = 1
while os.path.exists(OUTDIR):
    OUTDIR = f'{_base}_{_counter}'
    _counter += 1

os.makedirs(OUTDIR, exist_ok=True)
img = Image.open(SRC).convert('RGBA')
arr = np.array(img)
alpha = arr[:,:,3]
# stricter threshold to reduce anti-aliased specks
mask = alpha >= 128
h, w = mask.shape
visited = np.zeros((h,w), dtype=bool)
comps = []
nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
for y in range(h):
    for x in range(w):
        if mask[y,x] and not visited[y,x]:
            q = [(y,x)]
            visited[y,x] = True
            pts = []
            minx=maxx=x
            miny=maxy=y
            while q:
                cy,cx = q.pop()
                pts.append((cy,cx))
                if cx<minx: minx=cx
                if cx>maxx: maxx=cx
                if cy<miny: miny=cy
                if cy>maxy: maxy=cy
                for dy,dx in nb:
                    ny,nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx] = True
                        q.append((ny,nx))
            comps.append({'pts':pts,'bbox':(minx,miny,maxx,maxy),'area':len(pts)})

# keep meaningful components only
comps = [c for c in comps if c['area'] >= 1000]
comps.sort(key=lambda c:(c['bbox'][1], c['bbox'][0]))

manifest = []
for idx, c in enumerate(comps, 1):
    minx,miny,maxx,maxy = c['bbox']
    crop = arr[miny:maxy+1, minx:maxx+1].copy()
    # zero out other pixels in bbox
    local_mask = np.zeros((maxy-miny+1, maxx-minx+1), dtype=bool)
    for y,x in c['pts']:
        local_mask[y-miny, x-minx] = True
    crop[~local_mask] = [0,0,0,0]
    name = f'shape_{idx:02d}.png'
    out_path = os.path.join(OUTDIR, name)

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
    Image.fromarray(crop).save(out_path)

    manifest.append((name, c['area'], c['bbox']))

# contact sheet
thumbs = []
labels = []
for name, area, bbox in manifest:
    im = Image.open(os.path.join(OUTDIR, name)).convert('RGBA')
    bg = Image.new('RGBA', (220,220), (255,255,255,255))
    im.thumbnail((180,180))
    x = (220-im.width)//2
    y = 10 + (170-im.height)//2
    bg.alpha_composite(im, (x,y))
    # add simple border by painting edges
    a = np.array(bg)
    a[0,:,:3]=220; a[-1,:,:3]=220; a[:,0,:3]=220; a[:,-1,:3]=220; a[:,:,3]=255
    bg = Image.fromarray(a, 'RGBA').convert('RGB')
    thumbs.append(bg)
    labels.append(f"{name}\narea={area}")

# use pillow drawing for labels
from PIL import ImageDraw, ImageFont
font = ImageFont.load_default()
cell_w, cell_h = 220, 260
cols = 4
rows = math.ceil(len(thumbs)/cols) if thumbs else 1
sheet = Image.new('RGB', (cell_w*cols, cell_h*rows), 'white')
d = ImageDraw.Draw(sheet)
for i, thumb in enumerate(thumbs):
    r = i//cols
    c = i%cols
    x0 = c*cell_w
    y0 = r*cell_h
    sheet.paste(thumb, (x0, y0))
    d.text((x0+10, y0+225), labels[i], fill='black', font=font)

sheet.save(os.path.join(OUTDIR, 'contact_sheet.png'))
with open(os.path.join(OUTDIR, 'manifest.txt'), 'w', encoding='utf-8') as f:
    f.write(f'meaningful_components={len(manifest)}\n')
    for name, area, bbox in manifest:
        f.write(f'{name}\tarea={area}\tbbox={bbox}\n')
print(f'saved {len(manifest)} components to {OUTDIR}')
for row in manifest:
    print(row)