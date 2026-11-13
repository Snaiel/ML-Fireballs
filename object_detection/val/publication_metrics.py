"""Deterministic metrics for explicit tile/observation manifests (no inference)."""

import math
import re


SOURCE_ID = re.compile(r"^(\d+_\d{4}-\d{2}-\d{2}_\d{6}_(?:(?:[A-Za-z]|\d+)_)?DSC_\d+)(?=[._]|$)")


def source_frame_id(name):
    match = SOURCE_ID.match(name)
    if match is None:
        raise ValueError(f"Unrecognised source-frame identifier: {name}")
    return match.group(1)


def clip_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError(f"Invalid xyxy box: {box}")
    x1,y1,x2,y2 = map(float,box)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Reversed xyxy box: {box}")
    return [min(width,max(0,x1)), min(height,max(0,y1)),
            min(width,max(0,x2)), min(height,max(0,y2))]


def overlap(a, b, metric):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    if metric == 'intersects':
        return float(intersection > 0)
    area_a = max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    denominator = min(area_a,area_b) if metric=='iom' else area_a+area_b-intersection
    return intersection/denominator if denominator>0 else 0.


def match_boxes(ground_truth, predictions, metric='iom', threshold=0.5):
    """Maximum-cardinality, deterministic one-to-one matching at a fixed threshold.

    Matching is for fixed-operating-point counts, not ranked average precision.
    The caller must supply both sets in the same coordinate/visibility system.
    """
    if metric not in ('iom','iou','intersects'):
        raise ValueError('Unknown overlap metric')
    if not 0 < threshold <= 1:
        raise ValueError('Overlap threshold must be in (0,1]')
    edges = []
    for prediction in predictions:
        scores = [(overlap(truth,prediction,metric),i) for i,truth in enumerate(ground_truth)]
        edges.append([i for score,i in sorted(scores,key=lambda p:(-p[0],p[1])) if score>=threshold])
    owner = {}
    def augment(prediction, visited):
        for truth in edges[prediction]:
            if truth in visited:
                continue
            visited.add(truth)
            if truth not in owner or augment(owner[truth],visited):
                owner[truth] = prediction
                return True
        return False
    for prediction in range(len(predictions)):
        augment(prediction,set())
    return sorted((truth,prediction) for truth,prediction in owner.items())


def ratio(numerator,denominator):
    return numerator/denominator if denominator else None


def wilson(successes,total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes/total
    denominator=1+z*z/total
    middle=(p+z*z/(2*total))/denominator
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0,middle-half),min(1,middle+half)]


def score_samples(samples,metric='iom',threshold=0.5):
    """Count boxes/tiles/explicit observations; retain padding-only boxes as FPs.

    Boxes are unbordered xyxy. Both predictions and truth are clipped to the
    visible tile. Explicitly ineligible samples are reported without scoring.
    """
    observations,recovered = set(),set()
    seen = set()
    outcomes = []
    tp,fp,fn,positive,detected = 0,0,0,0,0
    excluded = []
    for sample in samples:
        identity=sample['sample_id']
        if identity in seen:
            raise ValueError(f'Duplicate sample ID: {identity}')
        seen.add(identity)
        if not sample.get('eligible',True):
            excluded.append(dict(sample_id=identity,reason=sample['exclusion_reason']))
            continue
        width,height=sample['width'],sample['height']
        truth=sample['ground_truth']
        truth_boxes=[clip_box(g['box'],width,height) for g in truth]
        if any(overlap(box,box,'iou')==0 for box in truth_boxes):
            raise ValueError(f'Ground truth has no visible area: {identity}')
        predicted=[clip_box(p['box'],width,height) for p in sample['predictions']]
        pairs=match_boxes(truth_boxes,predicted,metric,threshold)
        observations.update(g['observation_id'] for g in truth)
        recovered.update(truth[i]['observation_id'] for i,j in pairs)
        tp+=len(pairs); fp+=len(predicted)-len(pairs); fn+=len(truth)-len(pairs)
        positive+=bool(truth); detected+=bool(pairs)
        outcomes.append(dict(sample_id=identity,source_frame=sample['source_frame'],
            true_positives=len(pairs),false_positives=len(predicted)-len(pairs),
            false_negatives=len(truth)-len(pairs),matched_observations=sorted({truth[i]['observation_id'] for i,j in pairs})))
    summary=dict(evaluated_tiles=len(outcomes),excluded_tiles=len(excluded),positive_tiles=positive,
        recovered_positive_tiles=detected,tile_recovery=ratio(detected,positive),
        true_positive_boxes=tp,false_positive_boxes=fp,false_negative_boxes=fn,
        box_precision=ratio(tp,tp+fp),box_recall=ratio(tp,tp+fn),
        observations=len(observations),recovered_observations=len(recovered),
        observation_recovery=ratio(len(recovered),len(observations)),
        observation_recovery_wilson_95=wilson(len(recovered),len(observations)),
        missed_observations=sorted(observations-recovered),
        uncertainty_note='Wilson interval is descriptive binomial uncertainty; cross-camera/event dependence is not modelled.',
        matching=dict(metric=metric,threshold=threshold,algorithm='maximum cardinality one-to-one',visibility='both clipped to tile; padding-only predictions remain false positives'))
    return summary,outcomes,excluded
