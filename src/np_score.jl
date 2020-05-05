# np score as defined in http://www.stat.rice.edu/~cscott/pubs/npperform.pdf
"""
	np_score(fpr::Real, fnr::Real, α::Real)

Neyman-Pearson score for binary classifiers. The optimal classifier minimizes this. 
Note that in the paper, eq. (3), R0 = fpr, R1 = fnr.
"""
np_score(fpr::Real, fnr::Real, α::Real) = 1/α*max(0, fpr-α) + fnr

"""
	np_score(scores::Vector, y_true::Vector, α::Real)

Compute NP score for a vector of scores and labels.
"""
function np_score(scores::Vector, y_true::Vector, α::Real)
    # remember that y_i = 1 if score_i >= threshold
    ys_pred = map(x->predict_labels(scores,x), scores)
    fprs = map(y->false_positive_rate(y_true, y), ys_pred)
    fnrs = map(y->false_negative_rate(y_true, y), ys_pred)
    map(x->np_score(x[1], x[2], α), zip(fprs, fnrs))
end