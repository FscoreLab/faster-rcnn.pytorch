import numpy as np


def nms_py(dets, thresh):
    """Pure Python NMS baseline."""
    dets = dets.cpu().numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    # sort the detections according to scores, high to low values.

    keep = []

    # keep doing until all bboxes are covered by THIS BBOX
    while order.size > 0:
        # choose the bbox with highest iou score, namely THIS BBOX
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # filter out all detections which have IOUs larger than the threshold, which means they are covered by THIS BBOX
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
