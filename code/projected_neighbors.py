"""NumPy-only projected-neighbor scores; indexes contain training candidates.

Question identifiers are used only for leave-own-question-out training-side
calibration. Ordinary prediction supplies no exclusion identifiers, so input IDs
cannot determine the result. All learned quantities reside in the supplied asset.
"""
import numpy as np


def centered_embeddings(embeddings, valid):
    x=np.asarray(embeddings,dtype=np.float64)
    v=np.asarray(valid,dtype=bool)
    mean=(x*v[...,None]).sum(axis=1,keepdims=True)/np.maximum(v.sum(axis=1,keepdims=True)[...,None],1)
    return x-mean


def project(embeddings, valid, asset):
    return (centered_embeddings(embeddings,valid)-asset["projection_mean"]) @ asset["projection_rotations"]


def neighbor_logits(query, asset, ks=(7,21,63), exclude_questions=None, batch_rows=192):
    """Inverse Euclidean distance-weighted binary probability, clipped then logit.

    A stable distance sort resolves exact ties in stored training-row order. The
    optional exclusion array is an internal calibration tool, never input IDs.
    """
    query=np.asarray(query,dtype=np.float64)
    index=np.asarray(asset["index_vectors"],dtype=np.float64)
    labels=np.asarray(asset["index_labels"],dtype=np.float64)
    question=np.asarray(asset["index_question"],dtype=np.int64)
    ks=tuple(int(k) for k in ks)
    if query.ndim!=2 or index.ndim!=2 or query.shape[1]!=index.shape[1]:
        raise ValueError("Incompatible projected coordinates")
    if len(index)<max(ks) or min(ks)<1:raise ValueError("Neighbor count exceeds index")
    if exclude_questions is not None and len(exclude_questions)!=len(query):raise ValueError("Exclusion shape mismatch")
    result=np.zeros((len(query),len(ks)),dtype=np.float64)
    norm=(index*index).sum(axis=1)
    for start in range(0,len(query),batch_rows):
        x=query[start:start+batch_rows]
        distance=np.maximum((x*x).sum(axis=1)[:,None]+norm[None,:]-2*x@index.T,0.)
        if exclude_questions is not None:
            own=np.asarray(exclude_questions[start:start+len(x)],dtype=np.int64)
            distance[own[:,None]==question[None,:]]=np.inf
        nearest=np.argsort(distance,axis=1,kind="stable")[:,:max(ks)]
        chosen=np.take_along_axis(distance,nearest,axis=1)
        if not np.isfinite(chosen).all():raise ValueError("Insufficient nonexcluded neighbors")
        weights=1./np.maximum(np.sqrt(chosen),1e-3)
        numer=np.cumsum(weights*labels[nearest],axis=1)
        denom=np.cumsum(weights,axis=1)
        for column,k in enumerate(ks):
            probability=np.clip(numer[:,k-1]/denom[:,k-1],1e-4,1-1e-4)
            result[start:start+len(x),column]=np.log(probability)-np.log1p(-probability)
    return result


def score_candidates(embeddings, valid, asset, k):
    projected=project(embeddings,valid,asset)
    return neighbor_logits(projected.reshape(-1,projected.shape[-1]),asset,(k,))[:,0].reshape(len(projected),4)


def self_checks():
    index=np.arange(70,dtype=np.float64)[:,None]
    asset={"index_vectors":index,"index_labels":(np.arange(70)>=4).astype(np.float64),
           "index_question":np.arange(70)//4}
    query=np.array([[0.],[1.]])
    direct=neighbor_logits(query,asset,(7,21,63))
    excluded=neighbor_logits(query,asset,(7,21,63),np.zeros(2,dtype=np.int64))
    assert np.all(excluded>direct)
    assert np.array_equal(direct,neighbor_logits(query,asset,(7,21,63),batch_rows=1))
    assert neighbor_logits(np.empty((0,1)),asset,(7,)).shape==(0,1)
    raw=np.arange(24,dtype=np.float64).reshape(2,4,3);valid=np.ones((2,4),dtype=bool)
    perm=[2,0,3,1]
    assert np.array_equal(centered_embeddings(raw,valid)[:,perm],centered_embeddings(raw[:,perm],valid[:,perm]))
    return {"own_question_neighbors_excluded":True,"batch_independent":True,"empty":True,"candidate_centering_equivariant":True}
