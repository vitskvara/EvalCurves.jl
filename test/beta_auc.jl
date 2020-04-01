@testset "βAUC" begin
	Random.seed!(1234) # so the test are not totaly unpredictable
	
	n = 1000
	scores = rand(n)
	y_true = rand([0,1], n)
	d = 0.5
	fpr = 0.1
	ts = map(i->EvalCurves.threshold_at_fpr_sample(scores, y_true, fpr, d), 1:10)
	# test that they are not all the same
	@test !all(ts.==ts[1])
	@test isnan(EvalCurves.threshold_at_fpr_sample(scores, y_true, NaN, d))
	
	ns = 1000
	fprs = EvalCurves.fpr_distribution(scores, y_true, fpr, ns)
	@test length(fprs) == ns
	@test abs(fpr - mean(fprs)) < 0.01
	fprs = EvalCurves.fpr_distribution(scores, y_true, 0.001, ns; warns=false)
	@test any(isnan.(fprs))

	# this is based on the fact that beta(1,1) is uniform distribution
	x = rand(10000)
	α, β = EvalCurves.estimate_beta_params(x)   
	@test abs(α-1) < 0.02
	@test abs(β-1) < 0.02

	# x^(α-1)*(1-x)^(β-1)*Γ(α+β)/Γ(α)*Γ(β)
	@test exp(EvalCurves.beta_logpdf(0.5, 1, 1)) == 1.0
	@test exp(EvalCurves.beta_logpdf(0.5, 2, 1)) == 0.5*2
	@test exp(EvalCurves.beta_logpdf(0.2, 3, 2)) == 0.2^2*0.8*24/2

	# localized auc works the same way as βAUC, but the weights are uniform
	# between the minimum and maximum of the histogram
	fpr = 0.5
	lauc = EvalCurves.localized_auc(scores, y_true, fpr, ns)
	laucn = EvalCurves.localized_auc(scores, y_true, fpr, ns, normalize=true)
	@test lauc<laucn
	for fpr in 0:0.1:1.0
		lauc = EvalCurves.localized_auc(scores, y_true, fpr, ns, normalize=true)
		@test abs(lauc - fpr) < 0.05
	end
	
	# βAUC
	# since the score is random, roc is almost a diagonal and βAUC = fpr
	# but this stops working for fpr = 1.0
	for fpr in 0:0.1:0.9
		bauc = EvalCurves.beta_auc(scores, y_true, fpr, ns)
		@test abs(bauc - fpr) < 0.05
	end

	# if the roc is flat somewhere, the weighing by beta pdf should produce the same value as tpr@
	x = [0.0, 0.2, 0.2, 0.25, 0.25, 0.5, 0.5, 0.7, 0.9, 1.0]
	y = [0.0, 0.0, 0.25, 0.375, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
	fpr = 0.8
	fprs = EvalCurves.fpr_distribution(scores, y_true, fpr, ns)
	α, β = EvalCurves.estimate_beta_params(fprs)   
	# this is from beta_auc
	roci = EvalCurves.linear_interpolation(x,y, n=1001)
	dx = (roci[1][2] - roci[1][1])/2
	xw = roci[1][1:end-1] .+ dx
	w = exp.(EvalCurves.beta_logpdf.(xw, α, β))
	@test auc(roci..., w) ≈ EvalCurves.tpr_at_fpr(x,y,fpr) ≈ 1.0

	# linterp
	roc = roccurve(scores, y_true)
	interp_len = max(1001, length(roc[1]))
	roci = EvalCurves.linear_interpolation(roc..., n=interp_len)
	@test length(roci[1]) == length(roci[2]) == interp_len
	@test abs(auc(roc) - auc(roci)) < 0.001

	Random.seed!() # restart it again
end