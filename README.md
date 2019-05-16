# EvalCurves.jl
Lightweight implementation of PR and ROC curves.

Example of usage:

Obtain true and false positive rates from a vector of true labels and anomaly scores.

```julia
labels = [0; 0; 0; 0; 1]
ascores = [0.1; 0.2; 0.3; 0.5; 0.4]
fprvec, tprvec = EvalCurves.roccurve(ascores, labels)
```

Compute the are under ROC curve.

```julia
auroc = EvalCurves.auc(fprvec, tprvec)
```

!!! This is now deprecated !!!
Plot an ROC curve.

```julia
EvalCurves.plotroc((fprvec, tprvec, "$(auroc)"))
show()
```
