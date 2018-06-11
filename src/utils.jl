"""
    roccurve(ascorevec, labels)

Returns the roc curve data - true positive rate and false positive rate,
computed from anomaly score and labels vectors.
"""
function roccurve(ascorevec, labels)
    N = size(labels,1)
    @assert N == size(ascorevec,1)
    if isnan(ascorevec[1])
        warn("Anomaly score is NaN, check your inputs!")
    end
    fprvec = Array{Float,1}(N+2)
    tprvec = Array{Float,1}(N+2)
    p = sum(labels)
    n = N - p
    fpr = 1.0
    tpr = 1.0
    fprvec[1] = fpr # fp/n
    tprvec[1] = tpr # tp/p
    sortidx = sortperm(ascorevec)
    for i in 2:(N+1)
        (labels[sortidx[i-1]] == 0)? (fpr = fpr - 1/n) : (tpr = tpr -1/p)
        if (fpr <= tpr)
            fprvec[i] = fpr
            tprvec[i] = tpr
        else
            fprvec[i] = 1-fpr
            tprvec[i] = 1-tpr
        end
    end
    
    # ensure zeros
    tprvec[end] = 0.0
    fprvec[end] = 0.0
    
    # sort them
    isort = sortperm(fprvec)
    tprvec = tprvec[isort]
    fprvec = fprvec[isort]
    
    # avoid regression
    for i in 2:(N+2)
        if tprvec[i] < tprvec[i-1]
            tprvec[i] = tprvec[i-1]
        end
    end
    
    return round.(fprvec,10), round.(tprvec,10)
end

"""
    auc(x,y, [weights])

Computes the are under curve (x,y).
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
