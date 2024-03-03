"""
Modified from: https://github.com/tteepe/EarlyBird/blob/main/EarlyBird/evaluation/mot_bev.py
"""
import motmetrics as mm
import numpy as np


def mot_metrics_pedestrian(t, gt):
    # t and gt are in the form of [frame, id, x, y]
    acc = mm.MOTAccumulator()
    for frame in np.unique(gt[:, 0]).astype(int):
        gt_dets = gt[gt[:, 0] == frame]
        t_dets = t[t[:, 0] == frame]

        # format: gt, t; index 2 and 3 are the [x,y]
        C = mm.distances.norm2squared_matrix(gt_dets[:, 2:4]  * 0.025 , t_dets[:, 2:4]  * 0.025, max_d2=1)
        C = np.sqrt(C)

        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(),
                   C,
                   frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
    return summary
