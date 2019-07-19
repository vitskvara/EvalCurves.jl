# EvalCurves.jl
Lightweight implementation of PR, ROC curves and other binary classification metrics.

Example of usage:

Obtain true and false positive rates from a vector of true labels and anomaly scores.

```julia
using EvalCurves
labels = [0; 0; 0; 0; 1]
scores = [0.1; 0.2; 0.3; 0.5; 0.4]
fprvec, tprvec = EvalCurves.roccurve(scores, labels)
```

Compute the are under ROC curve.

```julia
auroc = EvalCurves.auc(fprvec, tprvec)
```

Do the same for the precision-recall curve.

```julia
labels = [0; 0; 0; 0; 1]
scores = [0.1; 0.2; 0.3; 0.5; 0.4]
recvec, precvec = EvalCurves.prcurve(scores, labels)
auprc = EvalCurves.auc(recvec, precvec)
```

Other binary classification metrics are available, this is just a small sample.

```julia
labels_true = [0; 0; 0; 1; 1]
labels_pred = [0; 0; 1; 0; 1]

# number of true positives
tp = EvalCurves.true_positive(labels_true, labels_pred)

# fpr = fp/n
fpr = EvalCurves.false_positive_rate(labels_true, labels_pred)

# acc = (tp+tn)/(p+n)
acc = EvalCurves.accuracy(labels_true, labels_pred) 

# mathews correlation coefficient
mcc = EvalCurves.matthews_correlation_coefficient(labels_true, labels_pred)
```

Some more complex metrics can be computed as well.

```julia
labels = [0; 0; 0; 1; 1]
scores = [0.1, 0.2, 0.4, 0.3, 0.5]

# precision at k most anomalous samples
EvalCurves.precision_at_k(scores, labels, 3)

# true positive rate @ p% false positive rate
fprvec, tprvec = EvalCurves.roccurve(scores, labels)
tpr50 = EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.5)

# returns a classifier threshold at given p% false positive rate.
t50 = EvalCurves.threshold_at_fpr(scores, labels, 0.5)

# estimates the volume of the decision space where classifier marks samples as normal using MC sampling
X = randn(2,100)
score_fun(x) = vec(sqrt.(sum(x.^2,dims=1)))
Xbounds = EvalCurves.estimate_bounds(X) # this returns the boundaries of the dataset
# everything larger than a threshold is an anomaly -> toghether with the chosen 
# anomaly score function, this will label all samples with positive second dimension as anomalies
for threshold in [0.0, 0.1, 0.5, 1.0, 2.0]
	vol = EvalCurves.volume_at_threshold(threshold, Xbounds, score_fun, 10000)
	println("$threshold := $vol")
end
```

