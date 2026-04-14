from PIL import Image
import numpy as np
from collections import Counter

img = Image.open('input/stadium_test1.png').convert('RGBA')
arr = np.array(img)
alpha = arr[:,:,3]
mask = alpha > 10
h, w = mask.shape
visited = np.zeros_like(mask, dtype=bool)
components = []
# 8-connectivity
neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
for y in range(h):
    for x in range(w):
        if mask[y,x] and not visited[y,x]:
            stack = [(y,x)]
            visited[y,x] = True
            pixels = []
            minx=maxx=x
            miny=maxy=y
            while stack:
                cy,cx = stack.pop()
                pixels.append((cy,cx))
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for dy,dx in neighbors:
                    ny,nx = cy+dy, cx+dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx] = True
                        stack.append((ny,nx))
            components.append({
                'bbox': (minx, miny, maxx, maxy),
                'count': len(pixels),
            })

print('components', len(components))
for i, comp in enumerate(sorted(components, key=lambda c: c['count'], reverse=True), 1):
    print(i, comp['count'], comp['bbox'])