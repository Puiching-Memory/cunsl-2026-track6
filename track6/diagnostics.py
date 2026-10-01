"""Label-oracle bounds for a fixed validation shortlist, never model scores."""
import math

import numpy as np


def shortlist_bounds(relevance, cutoffs=(10,50,100), denominator='min_relevant_k'):
    """Optimal top-10 ordering using only candidates within each fixed prefix.

    A label oracle moves all available positives to the front; unavailable
    positives still count in the AP denominator. This says nothing about
    whether an image-only reranker can realize that ordering.
    """
    rel=np.asarray(relevance)
    if rel.ndim!=2 or rel.dtype!=np.bool_ or not len(rel) or rel.shape[1]<10:
        raise ValueError('A nonempty boolean full-gallery relevance matrix is required')
    if denominator not in {'min_relevant_k','all_relevant'}:
        raise ValueError('Unknown AP denominator')
    if not cutoffs or len(set(cutoffs))!=len(cutoffs) or any(
        isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 10<=k<=rel.shape[1] for k in cutoffs):
        raise ValueError('Unique integer shortlist sizes between 10 and gallery size required')
    totals=rel.sum(axis=1)
    if (totals==0).any():
        raise ValueError('Every query needs a known positive in the full gallery')
    divisor=np.minimum(totals,10) if denominator=='min_relevant_k' else totals
    average=lambda values:math.fsum(float(v) for v in values)/len(values)
    top10=rel[:,:10]
    numerator=(top10*np.cumsum(top10,axis=1)/np.arange(1,11)).sum(axis=1)
    r1,ap=average(rel[:,0]),average(numerator/divisor)
    bounds={}
    for k in sorted(cutoffs):
        hits=rel[:,:k].sum(axis=1)
        recall=average(hits>0)
        upper_ap=average(np.minimum(hits,10)/divisor)
        bounds[str(k)]={'candidate_recall':recall,'positive_capture_fraction':average(hits/totals),
            'oracle_R@1':recall,'oracle_mAP@10':upper_ap,'oracle_direction_score':(recall+upper_ap)/2,
            'queries_with_no_positive':int((hits==0).sum()),
            'queries_missing_some_positive':int((hits<totals).sum())}
    return {'queries':len(rel),'gallery':rel.shape[1],'actual':{'R@1':r1,'mAP@10':ap,'score':(r1+ap)/2},
            'ap_denominator':denominator,'shortlists':bounds}
