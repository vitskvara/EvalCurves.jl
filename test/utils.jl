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
	@test round(aauroc; digits = 4)==1.0833
	caauroc = EvalCurves.auc(fprvec, tprvec, "centered")
	@test round(caauroc; digits = 4)==1.3524
	
	# partial auc test
	n = 200
	s = rand(n)
	y = rand([0,1], n)
	roc = roccurve(s, y)
	prs = [0.1, 0.24, 0.5, 0.8, 1.0]
	paucs = map(x->EvalCurves.partial_auc(roc..., x), prs)
	@test paucs[end] ≈ auc(roc...)
	pls = [0.0, 0.1, 0.24, 0.5, 0.8]
	paucs = map(x->EvalCurves.partial_auc(roc..., x[1], x[2]), zip(prs, pls))
	@test sum(paucs) ≈ auc(roc...)

	# simpler test
	tpr = [0.0, 0.2, 0.2, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]
	fpr = [0.0, 0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
	auroc = EvalCurves.auc(fpr, tpr)
	@test abs((EvalCurves.partial_auc(fpr,tpr,0.8) + 0.2) - auroc) < 1e-64
	@test abs(0.04 + EvalCurves.partial_auc(fpr,tpr,0.8,0.2) + 0.2 - auroc) < 1e-64

	# auc_at_p test
	@test abs((EvalCurves.auc_at_p(fpr,tpr,0.8) + 0.2) - auroc) < 1e-64

	# binarize
	@test EvalCurves.binarize([-1,-1,1,1,-1]) == [0,0,1,1,0]
	@test EvalCurves.binarize([2,2,2,2,2]) == [0,0,0,0,0]

	# other basic measures
	y = vcat(fill(0,1000),fill(1,500))
	yhat = vcat(fill(0,750),fill(1,500),fill(0,250))
	@test EvalCurves.true_positive(y,yhat)==250
	@test EvalCurves.true_negative(y,yhat)==750
	@test EvalCurves.false_positive(y,yhat)==250
	@test EvalCurves.false_negative(y,yhat)==250

	@test EvalCurves.true_positive_rate(y,yhat)==0.500
	@test EvalCurves.true_negative_rate(y,yhat)==0.7500
	@test EvalCurves.false_positive_rate(y,yhat)==0.2500
	@test EvalCurves.false_negative_rate(y,yhat)==0.500
	
	@test EvalCurves.precision(y,yhat)==250/(250+250)==0.5
	@test EvalCurves.accuracy(y,yhat)==(250+750)/(1000+500)==2/3
	@test EvalCurves.negative_predictive_value(y, yhat)==0.75
	@test EvalCurves.false_discovery_rate(y, yhat)==0.5
	@test EvalCurves.false_omission_rate(y, yhat)==0.25
	@test EvalCurves.f1_score(y, yhat)==0.5
	@test EvalCurves.mcc(y, yhat)==0.5

	# advanced measures
	# prec@k
	score = collect(1:10)/10
	y_true = [0,0,0,0,0,1,1,1,1,1]
	k=5
	@test EvalCurves.precision_at_k(score, y_true, k)==1
	y_true[10] = 0
	@test EvalCurves.precision_at_k(score, y_true, k)==0.8
	# tpr@fpr
	fpr = collect(0:10)/10
	tpr = sqrt.((collect(0:10)/10))
	p=0.05
	@test EvalCurves.tpr_at_fpr(fpr, tpr, p) == (tpr[1]+tpr[2])/2
	@test EvalCurves.tpr_at_fpr(fpr, tpr, 1.0) == tpr[end]
	
	# prec@rec
	s = [0.1, 0.4, 0.3, 0.5, 0.5, 0.6]
	l = [0, 0, 1, 0, 0, 1]
	rec, pr = EvalCurves.prcurve(s, l)
	@test EvalCurves.prec_at_rec(rec, pr, 1) == pr[end]
	@test EvalCurves.prec_at_rec(rec, pr, 0.5) == 0.25
	@test EvalCurves.prec_at_rec(rec, pr, 0.75) == 0.325	
	
	# threshold@fpr
	score = collect(1:10)/10
	y_true = [0,0,0,0,0,1,1,1,1,1]
	fpr=0.3
	@test EvalCurves.threshold_at_fpr(score, y_true, fpr) == 0.45
	@test isnan(EvalCurves.threshold_at_fpr(score, y_true, NaN))
	# volume estimates		
	Random.seed!(1234)
	X = randn(2,1000)
	score_fun(x) = vec(sqrt.(sum(x.^2,dims=1)))
	Xbounds = EvalCurves.estimate_bounds(X) 
	threshold = 0.5
	@test 0.01 < EvalCurves.sample_volume(score_fun, threshold, Xbounds) < 0.04
	pred_fun(X) = (score_fun(X) .> threshold)
	@test 0.01 < EvalCurves.sample_volume(pred_fun, Xbounds) < 0.04
	@test 0.01 < EvalCurves.volume_at_threshold(threshold, Xbounds, score_fun) < 0.04
	
	# now test the one with labels
	y_true = Int.(vec(score_fun(X).>1.2))
	fpr = 0.78
	threshold = EvalCurves.threshold_at_fpr(vec(score_fun(X)), y_true, fpr)
	@test 0.01 < EvalCurves.volume_at_fpr(fpr, Xbounds, score_fun, X, y_true) < 0.1
	Random.seed!()
end

@testset "threshold@FPR" begin
    y_true = [0,1,0,0,1]
	scores = [0.2, 0.4, 0.4, 0.4, 0.6]
	fpr = 0.5
	t = EvalCurves.threshold_at_fpr(scores, y_true, fpr)
	@test t ≈ 0.4
	@test fpr < EvalCurves.false_positive_rate(y_true, EvalCurves.predict_labels(scores, t)) 
	y_true = [0,1,0,0,0,1]
	scores = [0.2, 0.6, 0.4, 0.3, 0.5, 0.1]
	fpr = 0.5
	t=EvalCurves.threshold_at_fpr(scores, y_true, fpr)
	@test  t == 0.4
	@test fpr == EvalCurves.false_positive_rate(y_true, EvalCurves.predict_labels(scores, t)) 
	fpr = 0.25
	t=EvalCurves.threshold_at_fpr(scores, y_true, fpr)
	@test  t == 0.5
	@test fpr == EvalCurves.false_positive_rate(y_true, EvalCurves.predict_labels(scores, t)) 
	fpr = 0.75
	t=EvalCurves.threshold_at_fpr(scores, y_true, fpr)
	@test  t == 0.3
	@test fpr == EvalCurves.false_positive_rate(y_true, EvalCurves.predict_labels(scores, t)) 
end

@testset "degenerative case" begin
	x = ones(3)
	for y in [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
		fpr, tpr = roccurve(x, y)
		@test all(fpr .== [0.0, 1.0]) && all(tpr .== [0.0, 1.0])
		@test auc(fpr,tpr) == 0.5 
	end
end

@testset "AUPRC" begin
	ascores = [0.1, 0.4, 0.3, 0.5, 0.5, 0.6]
	labels = [0, 0, 1, 0, 0, 1]
	rec, pr = EvalCurves.prcurve(ascores, labels)
	rec2, pr2 = EvalCurves.prcurve(ascores, labels; zero_rec=true)
	@test length(rec2) == length(rec) + 1
end

@testset "f1@alpha" begin
	scores = [0.1, 0.4, 0.3, 0.5, 0.6]
	y_true = [0, 0, 1, 1, 1]
	@test EvalCurves.f1_score(y_true, [0,1,0,1,1]) == EvalCurves.f1_at_threshold(scores, y_true, 0.35)
	@test EvalCurves.f1_score(y_true, [0,1,0,1,1]) == EvalCurves.f1_at_threshold(scores, y_true, 0.4)
	@test EvalCurves.f1_score(y_true, [0,1,1,1,1]) == EvalCurves.f1_at_threshold(scores, y_true, 0.3)
	@test EvalCurves.f1_score(y_true, [0,1,1,1,1]) == EvalCurves.f1_at_fpr(scores, y_true, 0.5)
	@test EvalCurves.f1_score(y_true, [0,0,0,0,0]) == EvalCurves.f1_at_fpr(scores, y_true, 0)
	@test EvalCurves.f1_score(y_true, [1,1,1,1,1]) == EvalCurves.f1_at_fpr(scores, y_true, 1)
end

using PyCall
const sm = PyNULL()
runsklearntest = false
try
	copy!(sm, pyimport("sklearn.metrics"))
	global runsklearntest = true
catch e
	global 	runsklearntest = false
end

if !runsklearntest
	@info "SKlearn not found, skipping auc comparison tests."
else
	function compareskauc(labels, ascores)
		pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = false)
		pyauc = sm.auc(pyfpr, pytpr)

	 	fpr, tpr = EvalCurves.roccurve(ascores, labels)
	 	auc = EvalCurves.auc(fpr,tpr)

	 	@test pyauc ≈ auc
	end

	@testset "SKlearn AUROC comparison" begin
		# rand dataset
	 	for i in 1:30
	 		counts = rand(1:100, 2)
	 		labels = vcat(zeros(counts[1]), ones(counts[2]))
	 		ascores = rand(Float64, size(labels))

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

	    # exactly 0.5 AUC
	 	labels = [0; 0; 1; 1]
	 	ascores = [0; 1; 0; 1.]
	 	compareskauc(labels, ascores)
	end

	function compareskauprc(labels, ascores)
		pypr, pyrec, _ = sm.precision_recall_curve(labels, ascores)
		pyauprc = sm.auc(pyrec, pypr)

	 	rec, pr = EvalCurves.prcurve(ascores, labels; zero_rec=true)
	 	auprc = EvalCurves.auc(rec,pr)

	 	@test pyauprc ≈ auprc
	end


	@testset "SKlearn AUPRC comparison" begin
		# easy example
		ascores = [0.1, 0.4, 0.3, 0.5, 0.5, 0.6]
		labels = [0, 0, 1, 0, 0, 1]
		compareskauprc(labels, ascores)	

		# rand dataset
	 	for i in 1:30
	 		counts = rand(1:100, 2)
	 		labels = vcat(zeros(counts[1]), ones(counts[2]))
	 		ascores = rand(Float64, size(labels))

	 		compareskauprc(labels, ascores)
	 	end

	 	# Perfect dataset
	 	labels = vcat(zeros(50), ones(50))
	 	ascores = 1. .* labels
	 	compareskauprc(labels, ascores)

	 	# No true positive dataset
	 	labels = vcat(zeros(50), ones(50))
	 	ascores = -1. .* labels .+ 1
	 	compareskauprc(labels, ascores)

	    # exactly 0.5 AUC
	 	labels = [0; 0; 1; 1]
	 	ascores = [0; 1; 0; 1.]
	 	compareskauprc(labels, ascores)
		
	end
end