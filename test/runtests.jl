using EvalCurves
using Base.Test
import EvalCurves: auc, roccurve

# create artificial data
labels = [0; 0; 0; 0; 1]
ascores = [0.1; 0.2; 0.3; 0.5; 0.4]
N = size(labels,1)
@testset "ALL" begin
	fprvec, tprvec = EvalCurves.roccurve(ascores, labels)
	@test abs(sum(tprvec - [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))<1e-64
	@test abs(sum(fprvec - [0.0, 0.25, 0.25, 0.5, 0.75, 1.0]))<1e-64
	auroc = EvalCurves.auc(fprvec, tprvec)
	@test auroc==0.75
	aauroc = EvalCurves.auc(fprvec, tprvec, "1/x")
	@test round(aauroc, 4)==1.0833
	caauroc = EvalCurves.auc(fprvec, tprvec, "centered")
	@test round(caauroc, 4)==1.3524
	
	# auc_at_p test
	tpr = [0.0, 0.2, 0.2, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]
	fpr = [0.0, 0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
	auroc = EvalCurves.auc(fpr, tpr)
	@test ((EvalCurves.auc_at_p(fpr,tpr,0.8) + 0.2) - auroc) < 1e-64

end


@testset "degenerative case" begin
	x = ones(3)
	for y in [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
		fpr, tpr = roccurve(x, y)
		@test all(fpr .== [0.0, 1.0]) && all(tpr .== [0.0, 1.0])
		@test auc(fpr,tpr) == 0.5 
	end
end


using PyCall
@pyimport sklearn.metrics as sm

function compareskauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = false)
	pyauc = sm.auc(pyfpr, pytpr)

 	fpr, tpr = EvalCurves.roccurve(ascores, labels)
 	auc = EvalCurves.auc(fpr,tpr)

 	@test pyauc ≈ auc
end

@testset "Scikit learn comparison" begin

	# rand dataset
 	for i in 1:30
 		counts = rand(1:100, 2)
 		labels = vcat(zeros(counts[1]), ones(counts[2]))
 		ascores = rand(size(labels))

 		compareskauc(labels, ascores)
 	end

 	# Perfect dataset
 	labels = vcat(zeros(50), ones(50))
 	ascores = 1. .* labels
 	compareskauc(labels, ascores)

 	# No true positive dataset
 	labels = vcat(zeros(50), ones(50))
 	ascores = -1. .* labels .+ 1
 	compareskauc(labels, ascores)

#   # exactly 0.5 AUC
 	labels = [0; 0; 1; 1]
 	ascores = [0; 1; 0; 1.]
 	compareskauc(labels, ascores)
end

