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
            if sorted_scores[i] != sorted_scores[i - 1]
                fprvec[curveidx] = fpr
                tprvec[curveidx] = tpr
                curveidx += 1
            end
        else 
            tpr -= 1/p
        end
    end

    # ensure zeros
    curveidx += (tprvec[curveidx] == 0 && fprvec[curveidx] == 0) ? 0 : 1
    tprvec = tprvec[1:curveidx] 
    fprvec = fprvec[1:curveidx] 
    
    # sort them
    isort = sortperm(fprvec)
    tprvec = tprvec[isort]
    fprvec = fprvec[isort]
    
    # avoid regression
    for i in 2:length(tprvec)
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

auc(x::Tuple, weights = "same") = auc(x..., weights)

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
