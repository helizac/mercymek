"""Register video cameras against the generated Gaussian world with static SIFT correspondences.

This is CPU geometric fitting, not another image/depth network. Unsupported cameras retain the prior pose,
with the fallback recorded. It corrects short-window ego drift when static scene correspondences support it.
"""
import hashlib
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree


def register(case, g, road, result, rec, fallback, start, end):
    dst = case / 'environment' / 'cameras.npz'
    signature = hashlib.sha256((result['image_sha256'] + str(road) + str(start) + str(end) + 'sift-pnp-v3-road-height').encode()).hexdigest()
    if dst.exists():
        cache = np.load(dst)
        height = float(road.get('camera_height_m', 1.35))
        if str(cache['signature']) == signature and np.allclose(cache['matrices'][:,1,3], height):
            return cache['matrices'], cache['supported'], cache['errors']
    width = 1280
    image = cv2.imread(str(case / 'environment' / 'empty.png'))
    height = round(image.shape[0] * width / image.shape[1])
    factor = width / image.shape[1]
    image = cv2.resize(image, (width, height))
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.025)
    kp0, desc0 = sift.detectAndCompute(image, None)
    P = g['pos']
    good = (P[:, 2] > .5) & (P[:, 2] < 200) & (g['op'] > .6)
    P = P[good]
    focal = result['focal_px'] * factor
    uv = P[:, :2] / P[:, 2:] * focal + [width/2, height/2]
    tree = cKDTree(uv)
    distance, idx = tree.query(np.array([kp.pt for kp in kp0]), k=1)
    valid = distance < 2.5
    desc = desc0[valid]
    points = P[idx[valid]].astype(np.float64) * road['scale']
    meta = json.loads((case / 'meta.json').read_text())
    overlays = json.loads((case / 'overlays.json').read_text())
    basis = np.array(road['basis'])
    camera_height = road.get('camera_height_m',1.35)
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], float)
    matrices = np.array(fallback).copy()
    supported = np.zeros(len(matrices), bool)
    errors = np.full(len(matrices), np.nan)
    cap = cv2.VideoCapture(str(case / 'clip.mp4'))
    try:
        for k, t in enumerate(rec['times']):
            if not start <= t <= end:
                continue
            f = round(float(t) * meta['src_fps'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (width, height))
            mask = np.ones((height, width), np.uint8) * 255
            mask[round(height*.80):] = 0  # hood, wipers and timestamps are not world features
            for obj in overlays.get(str(f), {}).values():
                if obj['concept'] in ['license plate', 'face', 'sky']:
                    continue
                x1, y1, x2, y2 = np.array(obj['box']) * np.array([width/meta['width'], height/meta['height']]*2)
                mask[max(0,int(y1)-8):min(height,int(y2)+8),max(0,int(x1)-8):min(width,int(x2)+8)] = 0
            kp, desc_k = sift.detectAndCompute(frame, mask)
            if desc_k is None or len(desc) < 30:
                continue
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
            matches = matcher.knnMatch(desc, desc_k, k=2)
            matches = [a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance < .7*b.distance]
            if len(matches) < 30:
                continue
            p3 = points[[m.queryIdx for m in matches]]
            p2 = np.array([kp[m.trainIdx].pt for m in matches], np.float64)
            ok, rv, tv, inliers = cv2.solvePnPRansac(p3, p2, K, None, iterationsCount=500,
                reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 25 or len(inliers)/len(matches) < .25:
                continue
            take = inliers[:,0]
            ok, rv, tv = cv2.solvePnP(p3[take], p2[take], K, None, rv, tv, True)
            R = cv2.Rodrigues(rv)[0]
            C = -R.T @ tv[:,0]
            Cr = basis @ C + [0, camera_height, 0]
            if abs(Cr[1]-camera_height) > .8 or np.linalg.norm(Cr[[0,2]]) > 60:
                continue
            projected = cv2.projectPoints(p3[take], rv, tv, K, None)[0][:,0]
            error = float(np.median(np.linalg.norm(projected-p2[take],axis=1)))
            if error > 2.5:
                continue
            matrices[k,:3,:3] = basis @ R.T
            matrices[k,:3,3] = [Cr[0], camera_height, Cr[2]]
            supported[k], errors[k] = True, error
    finally:
        cap.release()
    np.savez_compressed(dst, matrices=matrices, supported=supported, errors=errors, signature=signature)
    (dst.with_suffix('.json')).write_text(json.dumps({'method':'static SIFT / SHARP 3D correspondences / PnP RANSAC',
        'supported_frames':int(supported.sum()), 'total_window_frames':int(((rec['times']>=start)&(rec['times']<=end)).sum()),
        'median_error_px':float(np.nanmedian(errors)) if supported.any() else None,
        'fallback':'prior VGGT poses where matching fails'},indent=2))
    return matrices, supported, errors
