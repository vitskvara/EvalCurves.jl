"""
    (fpr, tpr) = roccurve(score, labels)


    calculate false positive rate and true positive rate

"""
function roccurve(score::Vector, labels :: Vector)
    N = size(labels,1)
    @assert N == size(score,1)
    if isnan(score[1])
        warn("Anomaly score is NaN, check your inputs!")
    end
    fprvec = zeros(N+2)
    tprvec = zeros(N+2)
    p = sum(labels)
    n = N - p
    fpr = 1.0
    tpr = 1.0
    fprvec[1] = fpr # fp/n
    tprvec[1] = tpr # tp/p
    sortidx = sortperm(score)
    sorted_labels = labels[sortidx];
    sorted_scores = score[sortidx];
    curveidx = 2
    for i in 2:N
        if sorted_labels[i-1] == 0
            fpr -=  1/n
        else
            tpr -= 1/p
        end
        if sorted_scores[i] != sorted_scores[i - 1]
            fprvec[curveidx] = fpr
            tprvec[curveidx] = tpr
            curveidx += 1
        end
    end

    # ensure zeros
    curveidx += (tprvec[curveidx] == 0 && fprvec[curveidx] == 0) ? 0 : 1
    tprvec = tprvec[1:curveidx]
    fprvec = fprvec[1:curveidx]

    # sort out numerical -0
    fprvec = abs.(round.(fprvec; digits = 10))
    tprvec = abs.(round.(tprvec; digits = 10))

    # sort them
    isf = sortperm(fprvec)
    tprvec = tprvec[isf]
    fprvec = fprvec[isf]
    ist = sortperm(tprvec)
    tprvec = tprvec[ist]
    fprvec = fprvec[ist]

    # this creates a semi-concave envelope of the roc curve
    # experimental
    #if concave
    #    # this must be repeated at least N times
    #    for n in 1:N
    #        i = 1
    #        maxi = length(fprvec)-1
    #        while i <= maxi
    #        #for i in 2:(length(fprvec)-1)
    #            if (fprvec[i+1] == fprvec[i] && tprvec[i+1] > tprvec[i])
    #                maxi = length(fprvec)
    #                fprvec = fprvec[filter(x->x!=i,1:maxi)]
    #                tprvec = tprvec[filter(x->x!=i,1:maxi)]
    #                maxi = maxi - 2
    #            end
    #            i += 1
    #        end
    #    end
    #end

    return fprvec, tprvec
end

"""
    auc(x,y, [weights])

Computes the area under curve (x,y).
"""
function auc(x,y, weights = "same")
    # compute the increments
    dx = x[2:end] - x[1:end-1]
    dy = y[2:end] - y[1:end-1]

    if weights == "same"
        a = y[1:end-1] + dy/2
        b = dx
    elseif weights == "1/x"
        inz = x.!=0 # nonzero indices
        w = 1 ./x[inz]
        # w = w/sum(w) # this is numerically unstable
        a = (y[1:end-1] + dy/2)[inz[2:end]]
        a = a.*w
        b = dx[inz[2:end]]
    elseif weights == "centered"
        x = (x[2:end] + x[1:end-1])/2 # this is used for symmetric case
        inz = x.!=0 # nonzero indices
        w = 1 ./x[inz]
        a = (y[1:end-1] + dy/2)[inz]
        a = a.*w
        b = dx[inz]
    end

    return dot(a,b)
end

auc(x::Tuple, weights = "same") = auc(x..., weights)

"""
    auc_at_p(x,y,p,[weights,normalize])

Compute the left 100*p% of the integral under (x,y).
"""
function auc_at_p(x,y,p,weights="same";normalize=false)
    @assert 0.0 <= p <= 1.0
    # get the x up to which we integrate
    px = maximum(x)*p + minimum(x)*(1-p)
    # now, this is done so that the resulting (_x,_y)
    # under which is integrated ends correctly,
    # otherwise integrals will not sum up as part will be ommited
    inds = x.<=px
    py = tpr_at_fpr(x,y,px)
    # contruct the correct (_x,_y) and compute the new integral
    _x = push!(x[inds],px)
    _y = push!(y[inds],py)
    normalize ? (return auc(_x,_y,weights)/p) : (return auc(_x,_y,weights))
end

"""
    plotroc(args...)

Plot roc curves, where args is an iterable of triples (tprate, fprate, label).
"""
function plotroc(args...)
	f = figure()
	xlim([0,1])
	ylim([0,1])
	xlabel("false positive rate")
	ylabel("true positive rate")
	title("ROC")

    # plot the diagonal line
    plot(range(0,stop=1,length=100), range(0,stop=1,length=100),
        c = "gray", alpha = 0.5, label = "")
    for arg in args
        plot(arg[1], arg[2], label = arg[3], lw = 2)
    end
    legend()
end

###########################################
### basic binary classification metrics ###
###########################################

"""
   binarize(x)

Transform x to binary labels if needed.
"""
function binarize(x::Vector)
    vals = sort(unique(x))
    (length(vals) in [1,2]) ? nothing : error("values of x are not binary!")
    # if they are (false, true) or (0,1), do nothing
    if (vals == [0,1]) || (vals == [1]) || (vals == [0])
        return x
    # else relabel them
    else
        _x = copy(x)
        _x[_x.==vals[1]] == 0
        _x[_x.==vals[2]] == 1
        return _x
    end
end

true_positive_inds(y_true::Vector, y_pred::Vector) =
    (binarize(y_true).==1) .& (binarize(y_pred).==1)
true_negative_inds(y_true::Vector, y_pred::Vector) =
    (binarize(y_true).==0) .& (binarize(y_pred).==0)
false_positive_inds(y_true::Vector, y_pred::Vector) =
    (binarize(y_true).==0) .& (binarize(y_pred).==1)
false_negative_inds(y_true::Vector, y_pred::Vector) =
    (binarize(y_true).==1) .& (binarize(y_pred).==0)

true_positive(y_true, y_pred) = sum(true_positive_inds(y_true, y_pred))
true_negative(y_true, y_pred) = sum(true_negative_inds(y_true, y_pred))
false_positive(y_true, y_pred) = sum(false_positive_inds(y_true, y_pred))
false_negative(y_true, y_pred) = sum(false_negative_inds(y_true, y_pred))

"""
    true_positive_rate(y_true, y_pred)

Returns true positive rate (recall) = tp/(tp+fn).
"""
function true_positive_rate(y_true, y_pred)
    tp = true_positive(y_true, y_pred)
    return tp/(tp + false_negative(y_true, y_pred))
end

"""
    true_negative_rate(y_true, y_pred)

Returns true negative rate (specificity) = tn/(tn+fp).
"""
function true_negative_rate(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return tn/(tn + false_positive(y_true, y_pred))
end

"""
    false_positive_rate(y_true, y_pred)

Returns false positive rate (fallout) = fp/(fp+tn).
"""
function false_positive_rate(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return fp/(fp + true_negative(y_true, y_pred))
end

"""
    false_negative_rate(y_true, y_pred)

Returns false negative rate (miss rate) = fn/(fn+tp).
"""
function false_negative_rate(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return fn/(fn + true_positive(y_true, y_pred))
end

"""
    precision(y_true, y_pred)

Returns precision = tp/(tp+fp).
"""
function precision(y_true, y_pred)
    tp = true_positive(y_true, y_pred)
    return tp/(tp + false_positive(y_true, y_pred))
end

"""
    accuracy(y_true, y_pred)

Returns accuracy = (tp+tn)/(p+n).
"""
accuracy(y_true, y_pred) =
    (true_positive(y_true, y_pred) + true_negative(y_true, y_pred))/length(y_true)

"""
    negative_predictive_value(y_true, y_pred)

Returns negative predictive value = tn/(tn+fn).
"""
function negative_predictive_value(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return tn/(tn+false_negative(y_true, y_pred))
end

"""
   false_discovery_rate(y_true, y_pred)

Returns false false discovery rate = fp/(fp+tp).
"""
function false_discovery_rate(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return fp/(fp+true_positive(y_true, y_pred))
end

"""
    false_omission_rate(y_true, y_pred)

Returns false omission rate = fn/(fn+tn).
"""
function false_omission_rate(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return fn/(fn+true_negative(y_true, y_pred))
end

"""
    f1_score(y_true, y_pred)

Returns F1 score = 2*tp/(2*tp+fp+fn).
"""
function f1_score(y_true, y_pred)
    tp = true_positive(y_true, y_pred)
    return 2*tp/(2*tp+false_positive(y_true, y_pred)+false_negative(y_true, y_pred))
end

"""
    matthews_correlation_coefficient(y_true, y_pred)

Returns Matthews correlation coefficient = (tp*tn - fp*fn)/sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)).
"""
function matthews_correlation_coefficient(y_true, y_pred)
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return (tp*tn + fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
end

"""
    mcc(y_true, y_pred)

Returns Matthews correlation coefficient = (tp*tn - fp*fn)/sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)).
"""
mcc(y_true, y_pred) = matthews_correlation_coefficient(y_true, y_pred)

###############################
### advanced metrics for ad ###
###############################

"""
   precision_at_k(y_true, y_pred, score, k)

Precision at k most anomalous samples.
"""
function precision_at_k(y_true, y_pred, score::Vector, k::Int)
    lt, las = length(y_true), length(score)
    @assert lt == las
    @assert all(k .<= (lt, las))
    # sort anomaly scores from the largest
	isort = sortperm(scores, rev = true)
    return mean(y_true[isort][1:k])
end

"""
    tpr_at_fpr(fpr, tpr, p)

Return true positive rate @ p% false positive rate given
true positive rate and false positive rate vectors (ROC curve).
"""
function tpr_at_fpr(fpr::Vector, tpr::Vector, p::Real)
    @assert 0 <= p <= 1
    # find the place where p fals between two points at fpr
    inds = fpr.<=p
    lefti = sum(inds)
    righti = lefti + 1
    # now interpolate for tpr
    ratio = (p - fpr[lefti])/(fpr[righti] - fpr[lefti])
    return tpr[righti]*ratio + tpr[lefti]*(1-ratio)
end

"""
    threshold_at_fpr(y_true, score::Vector, p::Real)

Returns a classifier threshold at given p% false positive rate.
"""
function threshold_at_fpr(y_true, score::Vector, p::Real)
    N = length(score)
    @assert N == length(y_true)
    @assert 0.0 <= p .< 1.0

	descendingidx = sortperm(scores, rev = true)
    scores = scores[descendingidx]
    y_true = y_true[descendingidx]

    distincvalueidx = findall(diff(scores) .!= 0)
    thresholdidx = vcat(distincvalueidx, length(y_true))

    tps = cumsum(y_true)[thresholdidx]
    fps = thresholdidx .- tps
    fps = fps ./ fps[end]

    thresholds = scores[thresholdidx]

    ids = fpr .>= fps
    lastsmaller = sum(ids)
    if lastsmaller == 0
        @warn "No score to estimate lower FPR than $(fps[1])"
        return NaN # thresholds[1]
    elseif lastsmaller == length(fps)
        @warn "No score to estimate higher FPR than $(fps[end])"
        return NaN # thresholds[end]
    end

	lefti = lastsmaller
	righti = lefti + 1
	# now interpolate for tpr
    ratio = (p - fprs[lefti])/(fprs[righti] - fprs[lefti])
    return sorted_score[righti]*ratio + sorted_score[lefti]*(1-ratio)
end

"""
	sample_volume(score_fun, threshold, bounds, samples::Int = 10000)

Samples one volume of the decision space where classifier marks samples as normal.
"""
function sample_volume(score_fun, threshold, bounds, samples::Int = 10000)
    s = hcat(map(v -> length(v) == 2 ? rand(samples) .* (v[2] - v[1]) .+ v[1] : v[rand(1:length(v), samples)], bounds)...)
    scores = score_fun(s)
    return count(scores .<= threshold) / samples
end

function sample_volume(predict_fun, bounds, samples::Int = 10000)
    s = hcat(map(v -> length(v) == 2 ? rand(samples) .* (v[2] - v[1]) .+ v[1] : v[rand(1:length(v), samples)], bounds)...)
    hits = sum(predict_fun(s))
    return 1. - hits / n_samples
end

"""
	mc_volume_estimate(sample_volume_fun, iter::Int = 10)

For added precision this samples the volume `iter` number of times and takes
an average of that.
"""
function mc_volume_estimate(sample_volume_fun, iter::Int = 10)
    volume = 0.
    for i in 1:iter
        volume += sample_volume_fun()
    end
    return volume / iter
end

"""
	estimate_bounds(X::Matrix, threshold = 0.05)

Estimates the bounds of a dataset. It sets maximum and minimum for continuous
variables and saves all unique values for discrete ones. The decision boundary
for continuous/discrete variable is #unique value/#samples <= `threshold`
"""
function estimate_bounds(X::Matrix, threshold = 0.05)
	bounds = []
	for i in 1:size(X, 1)
		if (length(unique(X[i, :])) / length(X[i, :])) <= threshold
			push!(bounds, unique(X[i, :]))
		else
			push!(bounds, vcat(minimum(X[i, :]), maximum(X[i, :])))
		end
	end
	return bounds
end

"""
    volume_at_fpr(X, y_true, p, n_samples, predict_fun, ascore_fun, setthreshold_fun)

Computes the volume of space for which the classifier marks samples as normal.
To achieve better precision, X should be a union of train and test set
"""
function volume_at_fpr(X::Matrix, y_true, p::Real, n_samples::Int, predict_fun, ascore_fun, setthreshold_fun)
    ascores = ascore_fun(X)
    threshold = threshold_at_fpr(y_true, ascores, p)
	if threshold == NaN
		return NaN
	end
    setthreshold_fun(threshold)
    return mc_volume_estimate(() -> sample_volume(predict_fun, estimate_bounds(X))
end

function volume_at_fpr(X::Matrix, y_true, p::Real, n_samples::Int, ascore_fun)
    ascores = ascore_fun(X)
    threshold = threshold_at_fpr(y_true, ascores, p)
	if threshold == NaN
		return NaN
	end
    return mc_volume_estimate(() -> sample_volume(ascore_fun, threshold, estimate_bounds(X))
end
