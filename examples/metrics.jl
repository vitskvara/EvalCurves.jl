using AnomalyDetection, EvalCurves

# get data
trData, tstData = AnomalyDetection.getdata("haberman", 0.8, ["easy", "medium"])
trX, trY = trData.data, trData.labels
tstX, tstY = tstData.data, tstData.labels

# setup the model
indim = size(trX,1)
latentdim = 4
hiddendim = 10
esize = [indim, hiddendim, hiddendim, latentdim]
dsize = esize[end:-1:1]
model = AEmodel(esize, dsize, tracked = true, verbfit = true)

# fit the model and set a threshold
AnomalyDetection.fit!(model, trX[:,trY.==0])
model.contamination = sum(trY)/length(trY)
AnomalyDetection.setthreshold!(model, trX)

# get testing label estimates
as = AnomalyDetection.anomalyscore(model, tstX)
hatY = AnomalyDetection.predict(model, tstX)
fpr, tpr = EvalCurves.roccurve(as, tstY)
auroc = EvalCurves.auc(fpr, tpr)

# plot ROC and test the new metrics
EvalCurves.plotroc((fpr,tpr,"$(auroc)"))
println("precision at k=10 most anomalous samples:")
println(EvalCurves.precision_at_k(as,tstY, 10))
println("true positive rate @ 0.05 false positive rate:")
println(EvalCurves.tpr_at_fpr(fpr,tpr,0.05))
println("normalized AUC @ 0.05 false positive rate:")
println(EvalCurves.auc_at_p(fpr,tpr,0.05,normalize=true))
println("Matthews correlation coefficient:")
println(EvalCurves.mcc(tstY,hatY))
println("")

### now lets try the MC volume estimate ###
bounds = EvalCurves.estimate_bounds(hcat(trX, tstX))
tstas = AnomalyDetection.anomalyscore(model, tstX)
trfpr, trtpr = EvalCurves.roccurve(tstas, tstY)
for fpri in [0.05, 0.1, 0.3, 0.5, 0.9]
	threshold = EvalCurves.threshold_at_fpr(tstas, tstY, fpri)
	# volume computing function
	vf() = EvalCurves.volume_at_fpr(threshold, bounds, 
						x->AnomalyDetection.predict(model,x), 
						x->setfield!(model, :threshold, x),
						10000)

	vol = vf() #EvalCurves.mc_volume_estimate(vf, 10)
	println("at $fpri false positive rate:")
	println("enclosed volume: $vol")
	yhat = AnomalyDetection.predict(model, tstX)
	tfpr = EvalCurves.false_positive_rate(tstY, yhat)
	println("false positive rate: $tfpr")
	println("decision threshold: $(model.threshold)")
	println("")
end
