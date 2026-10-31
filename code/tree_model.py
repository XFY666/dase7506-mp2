"""Small NumPy-only inference for numeric histogram-gradient-boosted trees."""
import numpy as np


def export_trees(model):
    result = {'baseline': np.array(model._baseline_prediction).ravel()}
    roots, parts, offset = [], [], 0
    dtype = [('value', '<f8'), ('feature', '<i4'), ('threshold', '<f8'),
             ('left', '<i4'), ('right', '<i4'), ('leaf', 'u1'), ('missing_left', 'u1')]
    for predictors in model._predictors:
        node = predictors[0].nodes
        if node['is_categorical'].any():
            raise ValueError('Only numeric features are supported')
        part = np.empty(len(node), dtype=dtype)
        for dest, source in [('value','value'),('feature','feature_idx'),('threshold','num_threshold'),
                             ('leaf','is_leaf'),('missing_left','missing_go_to_left')]:
            part[dest] = node[source]
        part['left'] = node['left'] + offset
        part['right'] = node['right'] + offset
        roots.append(offset); parts.append(part); offset += len(node)
    result.update(nodes=np.concatenate(parts), roots=np.array(roots,dtype=np.int32))
    return result


def tree_scores(x, asset):
    x = np.asarray(x, dtype=np.float32)
    nodes = asset['nodes']
    out = np.full(len(x), float(asset['baseline'][0]), dtype=np.float64)
    for root in asset['roots']:
        active = np.arange(len(x))
        cursor = np.full(len(x), int(root), dtype=np.int32)
        while len(active):
            nd = nodes[cursor[active]]
            leaf = nd['leaf'].astype(bool)
            out[active[leaf]] += nd['value'][leaf]
            active = active[~leaf]
            if len(active):
                nd = nodes[cursor[active]]
                value = x[active, nd['feature']]
                left = np.where(np.isnan(value), nd['missing_left'].astype(bool), value <= nd['threshold'])
                cursor[active] = np.where(left, nd['left'], nd['right'])
    return out
