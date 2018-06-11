using EvalCurves
using Base.Test

# create artificial data
labels = [0; 0; 0; 0; 1]
ascores = [0.1; 0.2; 0.3; 0.5; 0.4]
N = size(labels,1)
@testset "ALL" begin
	fprvec, tprvec = EvalCurves.roccurve(ascores, labels)
	@test size(tprvec,1)==N+2
	@test size(fprvec,1)==N+2
	@test abs(sum(tprvec - [0, 0, 1, 1, 1, 1, 1]))<1e-64
	@test abs(sum(fprvec - [0, 0, 0.25, 0.5, 0.75, 0.75, 1]))<1e-64
	auroc = EvalCurves.auc(fprvec, tprvec)
	@test auroc==0.875
	EvalCurves.plotroc([(fprvec, tprvec, "test")]...)
	show()
	aauroc = EvalCurves.auc(fprvec, tprvec, "1/x")
	@test round(aauroc, 4)==1.5833
	caauroc = EvalCurves.auc(fprvec, tprvec, "centered")
	@test round(caauroc, 4)==2.3524
end
