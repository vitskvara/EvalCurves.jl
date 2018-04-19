"""
    roccurve(ascorevec, labels)

Returns the roc curve data computed from anomaly score and labels vectors.
"""
function roccurve(ascorevec, labels)
    N = size(labels,1)
    @assert N == size(ascorevec,1)
    fprvec = Array{Float,1}(N+2)
    recvec = Array{Float,1}(N+2)
    p = sum(labels)
    n = N - p
    fpr = 1.0
    rec = 1.0
    fprvec[1] = fpr # fp/n
    recvec[1] = rec # tp/p
    sortidx = sortperm(ascorevec)
    for i in 2:(N+1)
        (labels[sortidx[i-1]] == 0)? (fpr = fpr - 1/n) : (rec = rec -1/p)
        if (fpr <= rec)
            fprvec[i] = fpr
            recvec[i] = rec
        else
            fprvec[i] = 1-fpr
            recvec[i] = 1-rec
        end
    end
    
    # sort them
    isort = sortperm(fprvec)
    recvec = recvec[isort]
    fprvec = fprvec[isort]
    
    # avoid regression
    for i in 2:(N+2)
        if recvec[i] < recvec[i-1]
            recvec[i] = recvec[i-1]
        end
    end
    
    # ensure zeros
    recvec[1] = 0.0
    fprvec[1] = 0.0
    
    return recvec, fprvec
end

"""
    auc(x,y)

Computes the are under curve (x,y).
"""
function auc(x,y)
    # compute the increments
    dx = x[2:end] - x[1:end-1]
    dy = y[2:end] - y[1:end-1]
    
    return dot(y[1:end-1],dx) + dot(dx,dy)/2
end

"""
    plotroc(args...)

Plot roc curves, where args is an iterable of triples (fprate, tprate, label).
"""
function plotroc(args...)
    # plot the diagonal line
    p = plot(linspace(0,1,100), linspace(0,1,100), c = :gray, alpha = 0.5, xlim = [0,1],
    ylim = [0,1], label = "", xlabel = "false positive rate", ylabel = "true positive rate",
    title = "ROC")
    for arg in args
        plot!(arg[1], arg[2], label = arg[3], lw = 2)
    end
    return p
end
