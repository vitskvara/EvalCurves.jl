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
    fprvec = abs.(round.(fprvec,10))
    tprvec = abs.(round.(tprvec,10))

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
        w = 1./x[inz]
        # w = w/sum(w) # this is numerically unstable
        a = (y[1:end-1] + dy/2)[inz[2:end]]
        a = a.*w
        b = dx[inz[2:end]]
    elseif weights == "centered"
        x = (x[2:end] + x[1:end-1])/2 # this is used for symmetric case
        inz = x.!=0 # nonzero indices
        w = 1./x[inz]
        a = (y[1:end-1] + dy/2)[inz] 
        a = a.*w
        b = dx[inz]
    end
    
    return dot(a,b)
end

auc(x::Tuple, weights = "same") = auc(x..., weights)

"""
    auc_at_p(x,y,p,[weights])

Compute the left 100*p% of the integral under (x,y). 
"""
function auc_at_p(x,y,p,weights="same")
    @assert 0.0 <= p <= 1.0
    # get the x up to which we integrate
    px = maximum(x)*p + minimum(x)*(1-p)
    # now, this is done so that the resulting (_x,_y)
    # under which is integrated ends correctly, 
    # otherwise integrals will not sum up as part will be ommited
    inds = x.<=px
    py = tpr_at_p(x,y,px)
    # contruct the correct (_x,_y) and compute the new integral
    _x = push!(x[inds],px)
    _y = push!(y[inds],py)
    return auc(_x,_y,weights)
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
    plot(linspace(0,1,100), linspace(0,1,100), c = "gray", alpha = 0.5, label = "")
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
    (length(vals) in [1,2])? nothing : error("values of x are not binary!")
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
   precision_at_k(y_true, y_pred, ascores, k)

Precision at k most anomalous samples. 
"""
function precision_at_k(y_true, y_pred, ascores::Vector, k::Int)
    lt, lp, las = length(y_true), length(y_pred), length(ascores)
    @assert lt == lp == las
    @assert all(k.<= (lt,lp,las))
    # sort anomaly scores from the largest
    isort = sortperm(ascores,rev=true)
    return precision(y_true[isort][1:k], y_pred[isort][1:k])
end

"""
    tpr_at_p(fpr, tpr, p)

Return true positive rate @ p% false positive rate given
true positive rate and false positive rate vectors (ROC curve).
"""
function tpr_at_p(fpr::Vector, tpr::Vector, p::Real)
    @assert 0 <= p <= 1
    # find the place where p fals between two points at fpr
    inds = fpr.<=p
    lefti = sum(inds)
    righti = lefti + 1
    # now interpolate for tpr
    ratio = (p - fpr[lefti])/(fpr[righti] - fpr[lefti])
    return tpr[righti]*ratio + tpr[lefti]*(1-ratio)
end


