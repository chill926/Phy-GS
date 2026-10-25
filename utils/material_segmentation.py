"""CPU rules for hierarchical SAM3 instances and object-relative zoom windows."""
import numpy as np


def anchor_crop(box, image_size, options):
    width, height = image_size
    x0, y0, x1, y1 = box
    side = min(max((x1-x0)*options['width_factor'], (y1-y0)*options['height_factor'],
                   options['min_size']), width, height)
    left = np.clip((x0+x1-side)/2 + side*options.get('horizontal_offset', 0), 0, width-side)
    top = np.clip(y1-(1-options['bottom_padding'])*side, 0, height-side)
    return tuple(map(int, (round(left), round(top), round(left+side), round(top+side))))


def accept_instance(probabilities, score, group, full_shape, mask_threshold):
    area = int((probabilities >= mask_threshold).sum())
    full_pixels = int(np.prod(full_shape))
    return (np.isfinite(probabilities).all() and np.isfinite(score) and 0 <= score <= 1 and
            probabilities.min() >= 0 and probabilities.max() <= 1 and
            score >= group['instance_threshold'] and area >= group.get('min_mask_pixels', 8) and
            area <= full_pixels*group.get('max_mask_fraction', 1.) and
            area <= probabilities.size*group.get('max_crop_mask_fraction', 1.))


def compose_material_instances(instances, shape, mask_threshold, evidence_threshold, conflict_margin):
    """Specific instances can override parent/background masks, never their own missing pixels.

    Score comparisons are restricted to one priority level. An absent local detection
    cannot relabel anything. Opposite classes at equal priority retain the unknown option.
    """
    metal, nonmetal = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    priority = np.full(shape, -1, np.int16)
    for level in sorted({item['priority'] for item in instances}):
        maps = {'metal': np.zeros(shape, np.float32), 'nonmetal': np.zeros(shape, np.float32)}
        for item in instances:
            if item['priority'] != level:
                continue
            x0, y0, x1, y1 = item['window']
            probabilities = item['probabilities']
            evidence = np.where(probabilities >= mask_threshold, probabilities*item['score'], 0)
            target = maps[item['class']][y0:y1, x0:x1]
            np.maximum(target, evidence, out=target)
        present = (maps['metal'] >= evidence_threshold) | (maps['nonmetal'] >= evidence_threshold)
        metal[present], nonmetal[present] = maps['metal'][present], maps['nonmetal'][present]
        priority[present] = level
    labels = np.zeros(shape, np.uint8)
    labels[(metal >= evidence_threshold) & (metal-nonmetal >= conflict_margin)] = 2
    labels[(nonmetal >= evidence_threshold) & (nonmetal-metal >= conflict_margin)] = 1
    return labels, metal, nonmetal, priority
