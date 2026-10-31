"""Train answer-group objectives with nested question splits, using no test data.

Two representations isolate nonlinear scoring before versus after answer pooling.
Each configuration is selected on three-fold TRAIN predictions. Only the winner
is refitted and evaluated on validation. Inference contains no reference access.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from advanced_neural import development
from common import parse_answer
from experiment_utils import dump, metrics
from selection import select
from text_features import extract_features

OUT = Path('runs/answer_group_experiments')
FOLD_SEED = 20261015


def groups(bank):
    weights = np.zeros((len(bank), 4, 4), np.float32)
    for i, row in enumerate(bank):
        answers = {}
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            if answer is not None:
                answers.setdefault(answer, []).append(j)
        for k, ids in enumerate(answers.values()):
            weights[i, k, ids] = 1 / len(ids)
    return weights


def make_features(embeddings, text, weights, representation):
    membership = weights > 0
    count = membership.sum(axis=2)
    ng = (count > 0).sum(axis=1)
    support = np.stack((np.log(np.maximum(count, 1)),
                        count / np.maximum(count.sum(axis=1, keepdims=True), 1),
                        np.broadcast_to(ng[:, None], count.shape)), axis=2).astype(np.float32)
    if representation == 'candidate':
        candidate_support = np.einsum('ngc,ngd->ncd', membership, support)
        return np.concatenate((embeddings, text, candidate_support), axis=2).astype(np.float32)
    mean_e = np.einsum('ngc,ncd->ngd', weights, embeddings)
    mean_t = np.einsum('ngc,ncd->ngd', weights, text)
    maxima = np.zeros_like(mean_t)
    deviation = np.zeros_like(mean_t)
    for i in range(len(weights)):
        for g in range(4):
            ids = membership[i, g]
            if ids.any():
                maxima[i, g] = text[i, ids].max(axis=0)
                deviation[i, g] = text[i, ids].std(axis=0, dtype=np.float64)
    return np.concatenate((mean_e, mean_t, maxima, deviation, support), axis=2).astype(np.float32)


def fit_statistics(x, valid):
    rows = x[valid]
    mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(rows.std(axis=0, dtype=np.float64), 1e-5).astype(np.float32)
    return mean, scale


def normalized(x, mean, scale):
    return np.clip((x - mean) / scale, -12, 12).astype(np.float32)


def initialize(dim, hidden, seed=7506):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1 / np.sqrt(dim), (dim, hidden)).astype(np.float32),
            np.zeros(hidden, np.float32), rng.normal(0, 1 / np.sqrt(hidden), hidden).astype(np.float32),
            np.zeros(dim, np.float32)]


def scores(x, model):
    w1, b, w2, skip = model
    return np.tanh(x @ w1 + b) @ w2 + x @ skip


def group_objective(score, labels, valid, objective):
    if objective == 'ce':
        masked = np.where(valid, score, -np.inf)
        logz = logsumexp(masked, axis=1, keepdims=True)
        gradient = (np.exp(masked - logz) - labels) / len(score)
        loss = np.mean(logz[:, 0] - (score * labels).sum(axis=1))
    else:
        pairs = labels[:, :, None] * (valid & ~labels)[:, None, :]
        weight = pairs / np.maximum(pairs.sum(axis=(1, 2))[:, None, None], 1)
        difference = score[:, None, :] - score[:, :, None]
        loss = np.mean((np.logaddexp(0, difference) * weight).sum(axis=(1, 2)))
        probability = expit(difference) * weight / len(score)
        gradient = probability.sum(axis=1) - probability.sum(axis=2)
    return float(loss), gradient.astype(score.dtype)


def loss_gradient(x, weights, labels, valid, model, config, rng=None):
    w1, b, w2, skip = model
    hidden = np.tanh(x @ w1 + b)
    mask = 1
    if rng is not None and config['dropout']:
        mask = (rng.random(hidden.shape) >= config['dropout']).astype(hidden.dtype) / (1 - config['dropout'])
    used = hidden * mask
    raw = used @ w2 + x @ skip
    group_score = np.einsum('ngc,nc->ng', weights, raw) if config['representation'] == 'candidate' else raw
    loss, gradient = group_objective(group_score, labels, valid, config['objective'])
    if config['representation'] == 'candidate':
        gradient = np.einsum('ngc,ng->nc', weights, gradient)
    dh = gradient[:, :, None] * w2 * mask * (1 - hidden * hidden)
    grads = [x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, len(w2)), dh.sum(axis=(0, 1)),
             used.reshape(-1, len(w2)).T @ gradient.ravel(), x.reshape(-1, x.shape[-1]).T @ gradient.ravel()]
    for k in (0, 2, 3):
        loss += .5 * config['l2'] * float(np.sum(model[k] ** 2))
        grads[k] += config['l2'] * model[k]
    return loss, grads


def evaluation(x, weights, labels, valid, model, config):
    raw = scores(x, model)
    grouped = np.einsum('ngc,nc->ng', weights, raw) if config['representation'] == 'candidate' else raw
    loss, _ = group_objective(grouped, labels, valid, config['objective'])
    chosen = np.argmax(np.where(valid, grouped, -np.inf), axis=1)
    return loss, int(labels[np.arange(len(labels)), chosen].sum())


def fit(x, weights, labels, valid, config, *, heldout=None, epochs=80):
    model = initialize(x.shape[-1], config['hidden'])
    m = [np.zeros_like(p) for p in model]
    v = [np.zeros_like(p) for p in model]
    rng = np.random.default_rng(7507)
    curve, step, best, best_epoch, stale = [], 0, float('inf'), 0, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            ix = order[start:start + 128]
            _, gradients = loss_gradient(x[ix], weights[ix], labels[ix], valid[ix], model, config, rng)
            step += 1
            for k, g in enumerate(gradients):
                m[k] = .9 * m[k] + .1 * g
                v[k] = .999 * v[k] + .001 * g * g
                model[k] -= .001 * (m[k] / (1 - .9 ** step)) / (np.sqrt(v[k] / (1 - .999 ** step)) + 1e-8)
        record = {'epoch': epoch}
        if heldout is not None:
            loss, correct = evaluation(*heldout, model, config)
            record.update(loss=loss, correct=correct, n=len(heldout[0]))
            if loss < best - 1e-6:
                best, best_epoch, stale = loss, epoch, 0
            else:
                stale += 1
        curve.append(record)
        if heldout is not None and stale >= 10:
            break
    assert all(np.isfinite(p).all() for p in model)
    return model, curve, best_epoch if heldout is not None else epochs


def candidate_scores(raw, weights, representation):
    return raw if representation == 'candidate' else np.einsum('ngc,ng->nc', weights > 0, raw)


def scale_scores(raw, valid):
    return float(max(np.std((raw - raw.mean(axis=1, keepdims=True))[valid]), 1e-6))


def save_asset(path, model, mean, scale):
    np.savez(path, w1=model[0], b1=model[1], w2=model[2], skip=model[3], mean=mean, scale=scale)


def predict_scores(bank, arrays, config, asset):
    if not bank:
        return np.empty((0, 4), np.float32)
    weights = groups(bank)
    text = extract_features(bank)
    x = make_features(arrays['embeddings'], text, weights, config['representation'])
    model = [asset[k] for k in ('w1', 'b1', 'w2', 'skip')]
    return candidate_scores(scores(normalized(x, asset['mean'], asset['scale']), model), weights, config['representation'])


def gradient_check():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(2, 4, 3))
    weights = np.asarray([[[.5,.5,0,0],[0,0,.5,.5],[0,0,0,0],[0,0,0,0]],
                          [[1,0,0,0],[0,1,0,0],[0,0,.5,.5],[0,0,0,0]]], dtype=float)
    valid = weights.sum(axis=2) > 0
    labels = np.asarray([[1,0,0,0],[0,0,1,0]], dtype=bool)
    checks = []
    for representation in ('candidate', 'group'):
        for objective in ('ce', 'pairwise'):
            config = dict(representation=representation, objective=objective, dropout=0, hidden=5, l2=.01)
            model = [p.astype(float) for p in initialize(3, 5)]
            _, grad = loss_gradient(x, weights, labels, valid, model, config)
            error = 0.
            for k, idx in ((0,(1,2)),(1,(3,)),(2,(2,)),(3,(1,))):
                step = 1e-5
                model[k][idx] += step
                hi = loss_gradient(x, weights, labels, valid, model, config)[0]
                model[k][idx] -= 2 * step
                lo = loss_gradient(x, weights, labels, valid, model, config)[0]
                model[k][idx] += step
                error = max(error, abs((hi-lo)/(2*step) - grad[k][idx]))
            assert error < 1e-6, (representation, objective, error)
            checks.append(dict(representation=representation, objective=objective, max_error=error))
    return checks


def configurations():
    configs = []
    for representation in ('candidate', 'group'):
        for objective in ('ce', 'pairwise'):
            for dropout in (0., .2):
                configs.append(dict(representation=representation, objective=objective, dropout=dropout, hidden=128, l2=.01))
    for l2 in (.001, .03):
        configs.append(dict(representation='group', objective='ce', dropout=.2, hidden=128, l2=l2))
    for hidden in (64, 256):
        configs.append(dict(representation='candidate', objective='ce', dropout=.2, hidden=hidden, l2=.01))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-only', action='store_true')
    parser.add_argument('--validate-selected', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.validate_selected:
        chosen = json.loads((OUT/'selected_config.json').read_text())
        path = OUT/'validation.json'
        if path.exists():
            raise SystemExit('Validation record exists; refusing a repeated development evaluation.')
        bank, emb, labels, valid, hard = development('validation')
        with np.load(OUT/'selected.npz') as asset:
            logits = predict_scores(bank, {'embeddings':emb}, chosen['configuration'], asset)
        result = metrics(select(bank, logits, 'mean'), labels, hard)
        np.savez(OUT/'validation_logits.npz', raw=logits, normalized=logits/chosen['training_scale'])
        dump(path, {'metrics':result, 'selected_from_training_only':True, 'configuration':chosen['configuration']})
        print(result, flush=True)
        return
    if (OUT/'protocol.json').exists():
        raise SystemExit('Experiment records exist; will not silently repeat or replace the screen.')
    start = time.perf_counter()
    protocol = {'fold_seed':FOLD_SEED, 'folds':3, 'fold_mapping':'fold[rng.permutation(N)] = arange(N) % 3',
                'configurations':configurations(), 'train_only_selection':'hard OOF correct, then mean group CE, then smaller hidden size',
                'inner_stopping':'80/20 of outer-fit questions; fitting-side statistics; min heldout group objective, patience10/max80',
                'refit':'fresh same-seed model at selected epoch; outer fitting questions only',
                'final_refit':'median outer selected epochs, all training hard questions; full-training input statistics',
                'objective':'Direct correct answer group versus incorrect groups, question weighted equally; no easy questions in optimizer batches',
                'inference':'mean candidate group scores or nonlinear scoring after input group pooling; finite features and training statistics only',
                'gradient_checks':gradient_check(), 'threads':4, 'test_access':False,
                'validation':'Only selected model, separate command after recording selection',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(OUT/'protocol.json', protocol)
    bank, emb, labels, valid, hard = development('train')
    weights = groups(bank)
    gvalid = weights.sum(axis=2)>0
    glabels = np.einsum('ngc,nc->ng', weights, labels)>0
    assert np.all(glabels.sum(axis=1)==1)
    fold = np.empty(len(bank), np.int8)
    fold[np.random.default_rng(FOLD_SEED).permutation(len(bank))] = np.arange(len(bank))%3
    dump(OUT/'alignment.json', [{'id':row['id'], 'fold':int(f)} for row,f in zip(bank,fold)])
    text = extract_features(bank)
    features = {rep:make_features(emb,text,weights,rep) for rep in ('candidate','group')}
    outputs, results = {}, []
    with threadpool_limits(limits=4):
        for index, config in enumerate(configurations()):
            tick = time.perf_counter()
            key = f'config_{index:02d}'
            x = features[config['representation']]
            feature_valid = valid if config['representation']=='candidate' else gvalid
            oof = np.zeros((len(bank),4), np.float32)
            oof_normalized = np.zeros_like(oof)
            folds = []
            for f in range(3):
                fit_ids, eval_ids = np.where(fold!=f)[0], np.where(fold==f)[0]
                order = np.random.default_rng(FOLD_SEED+100+f).permutation(fit_ids)
                cutoff = int(.8*len(order))
                fit_inner, stop_inner = order[:cutoff], order[cutoff:]
                mean, scale = fit_statistics(x[fit_inner], feature_valid[fit_inner])
                ii, si = fit_inner[hard[fit_inner]], stop_inner[hard[stop_inner]]
                heldout = (normalized(x[si],mean,scale),weights[si],glabels[si],gvalid[si])
                _, curve, epoch = fit(normalized(x[ii],mean,scale),weights[ii],glabels[ii],gvalid[ii],config,heldout=heldout)
                mean, scale = fit_statistics(x[fit_ids],feature_valid[fit_ids])
                active = fit_ids[hard[fit_ids]]
                model, _, _ = fit(normalized(x[active],mean,scale),weights[active],glabels[active],gvalid[active],config,epochs=epoch)
                raw = candidate_scores(scores(normalized(x[eval_ids],mean,scale),model),weights[eval_ids],config['representation'])
                fit_raw = candidate_scores(scores(normalized(x[fit_ids],mean,scale),model),weights[fit_ids],config['representation'])
                score_std = scale_scores(fit_raw,valid[fit_ids])
                oof[eval_ids], oof_normalized[eval_ids] = raw, raw/score_std
                save_asset(OUT/f'{key}_fold{f}.npz',model,mean,scale)
                folds.append({'fold':f,'fit_questions':len(fit_ids),'eval_questions':len(eval_ids),'fit_hard':len(active),
                              'inner_fit_hard':len(ii),'inner_stopping_hard':len(si),'epoch':epoch,'scale':score_std,'curve':curve})
            grouped = np.einsum('ngc,nc->ng',weights,oof)
            ce = group_objective(grouped[hard],glabels[hard],gvalid[hard],'ce')[0]
            row = {'name':key,'configuration':config,'metrics':metrics(select(bank,oof,'mean'),labels,hard),
                   'oof_group_ce':ce,'seconds':time.perf_counter()-tick,'folds':folds}
            results.append(row)
            outputs[key] = oof
            outputs[key+'_normalized'] = oof_normalized
            dump(OUT/'results.json',results)
            print({k:v for k,v in row.items() if k!='folds'},flush=True)
        best = max(results,key=lambda r:(r['metrics']['hard_correct'],-r['oof_group_ce'],-r['configuration']['hidden']))
        config = best['configuration']
        epoch = int(np.median([r['epoch'] for r in best['folds']]))
        x = features[config['representation']]
        feature_valid = valid if config['representation']=='candidate' else gvalid
        mean,scale=fit_statistics(x,feature_valid)
        model,_,_=fit(normalized(x[hard],mean,scale),weights[hard],glabels[hard],gvalid[hard],config,epochs=epoch)
        final_raw=candidate_scores(scores(normalized(x,mean,scale),model),weights,config['representation'])
        save_asset(OUT/'selected.npz',model,mean,scale)
        selected={'configuration':config,'epochs':epoch,'training_scale':scale_scores(final_raw,valid),'selected_from':best['name'],
                  'train_oof_metrics':best['metrics'],'oof_group_ce':best['oof_group_ce'],'asset_file_bytes':(OUT/'selected.npz').stat().st_size}
        dump(OUT/'selected_config.json',selected)
        np.savez(OUT/'oof_logits.npz',raw=outputs[best['name']],normalized=outputs[best['name']+'_normalized'],fold=fold,labels=labels,valid=valid,hard=hard)
        np.savez(OUT/'all_oof_logits.npz',**outputs)
        dump(OUT/'summary.json',{'selected':selected,'research_model_fits':len(results)*6+1,'seconds':time.perf_counter()-start,
                                'validation_pending':True,'test_access':False})
        print(selected,flush=True)


if __name__=='__main__':
    main()
