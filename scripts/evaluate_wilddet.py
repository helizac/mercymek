"""Inspect original-frame WildDet cuboids and record consistency gates (not 3D accuracy)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config
from mercy.ingest.wilddet import decode, qualify


def evaluate(output):
    rows = json.loads((Path(output) / 'wilddet-predictions/results.json').read_text())
    destination = ROOT / 'data/research/wilddet3d/evaluation' / Path(output).name
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for row in rows:
        case = config.case_dir(row['case'])
        profiles = json.loads((case / 'vehicles/profiles.json').read_text())
        cap = cv2.VideoCapture(str(case / 'clip.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame']); ok, visual = cap.read(); cap.release()
        if not ok:
            raise ValueError('Cannot read original frame')
        accepted, rejected = [], []
        for prediction in row['predictions']:
            dims = profiles.get(prediction['track'], {}).get('dimensions_m')
            anchor, reason = qualify(prediction, row, np.array(dims) if dims else None)
            color = (70, 240, 90) if anchor else (60, 80, 240)
            a, b, c, d = map(int, prediction['source_box'])
            cv2.rectangle(visual, (a, b), (c, d), color, 1)
            label = prediction['track'] + (f" 3D:{prediction['score3d']:.2f}" if anchor else ' ' + reason)
            cv2.putText(visual, label, (max(0, a), max(40, b - 4)), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1, cv2.LINE_AA)
            if anchor:
                accepted.append(anchor)
                p = decode(prediction['box3d'], row['C'], row['R'])['camera_corners']
                xy = p @ np.asarray(row['K']).T; xy = xy[:, :2] / xy[:, 2:]
                xy = np.clip(xy, -10000, 10000).astype(int)
                for i in range(8):
                    for bit in [1, 2, 4]:
                        j = i ^ bit
                        if j > i:
                            cv2.line(visual, tuple(xy[i]), tuple(xy[j]), color, 1, cv2.LINE_AA)
            else:
                rejected.append(dict(track=prediction['track'], reason=reason))
        cv2.rectangle(visual, (0, 0), (visual.shape[1], 29), (10, 10, 10), -1)
        cv2.putText(visual, f'WildDet3D RGB | {row["case"]} frame {row["frame"]} | qualified {len(accepted)}/{len(row["predictions"])} | consistency gates, not 3D ground truth',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
        name = f'{row["case"]}-{row["frame"]:05d}'
        cv2.imwrite(str(destination / (name + '.jpg')), visual)
        record = summary.setdefault(row['case'], dict(frames=0, predictions=0, accepted=0, tracks=set(), rejected=Counter(), seconds=0., peak_gpu_bytes=0))
        record['frames'] += 1; record['predictions'] += len(row['predictions']); record['accepted'] += len(accepted)
        record['tracks'].update(a['track'] for a in accepted)
        record['rejected'].update(r['reason'] for r in rejected)
        record['seconds'] += row['seconds']; record['peak_gpu_bytes'] = max(record['peak_gpu_bytes'], row['peak_gpu_bytes'])
    for record in summary.values():
        record['tracks'] = sorted(record['tracks']); record['rejected'] = dict(record['rejected'])
    report = dict(output=str(Path(output).resolve()), cases=summary, runtime=rows[0]['runtime'],
        gates=dict(score2d_min=.3, score3d_min=.15, prompt_iou_min=.5, cuboid_reprojection_iou_min=.35,
            max_tilt_deg=25, ground_tolerance_m='max(.8, .55 * predicted height)', class_dimension_ratio=[.5, 1.85]),
        accuracy_claim='No human-labelled 3D ground truth; these are internal consistency checks only.')
    (destination / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    latest = json.loads((ROOT / 'data/kaggle-wilddet/latest.json').read_text()) if args.output is None else None
    evaluate(args.output or ROOT / latest['output'].replace('\\', '/'))
