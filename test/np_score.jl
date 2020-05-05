@testset "NP score" begin
	n = 200
	s = rand(n)
	y = rand([0,1], n)
	α = 0.1
	t = 0.3
	y_pred = EvalCurves.predict_labels(s, t)
	fpr = EvalCurves.false_positive_rate(y, y_pred)
	fnr = EvalCurves.false_negative_rate(y, y_pred)

	nps = EvalCurves.np_score(fpr, fnr, α)
	@test nps != 0
	@test isnan(EvalCurves.np_score(fpr, fnr, NaN))
	@test isnan(EvalCurves.np_score(NaN, fnr, α))
	@test isnan(EvalCurves.np_score(NaN, NaN, α))

	npss = EvalCurves.np_score(s, y, α)
	@test all(.!isnan.(npss))
	@test length(npss) == n
end