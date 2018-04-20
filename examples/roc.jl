push!(LOAD_PATH, "../src")
using EvalCurves

# true labels
labels = [0; 0; 0; 0; 1]
println("True labels: ", labels)
# estimated anomaly scores
ascores = [0.1; 0.2; 0.3; 0.5; 0.4]
println("Estimated anomaly scores: ", ascores)

# this returns a vector of true positive and false positive rates
tprvec, fprvec = EvalCurves.roccurve(ascores, labels)
println("True positive rates: ", tprvec)
println("False positive rates: ", fprvec)

# area under curve
auroc = EvalCurves.auc(fprvec, tprvec)
println("AUROC: $(auroc)")

# roc plot
display(EvalCurves.plotroc([(fprvec, tprvec, "test")]...))
Plots.gui()


