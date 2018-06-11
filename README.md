# EvalCurves.jl
Lightweight implementation of PR and ROC curves.

Example of usage:

Obtain true and false positive rates from a vector of true labels and anomaly scores.

`labels = [0; 0; 0; 0; 1]
ascores = [0.1; 0.2; 0.3; 0.5; 0.4]

fprvec, tprvec = EvalCurves.roccurve(ascores, labels)`

Compute the are under ROC curve.

`auroc = EvalCurves.auc(fprvec, tprvec)`

Plot an ROC curve.

`EvalCurves.plotroc([(fprvec, tprvec, "$(auroc)")]...)
show()`
