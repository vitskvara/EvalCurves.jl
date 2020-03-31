"""
    threshold_at_fpr_sample(scores::Vector, y_true::Vector, fpr::Real, d::Real)

Subsample the input `scores` to obtain a random threshold value for a given `fpr` value.
Argument `d` is the relative number of the subsampled scores.
"""
function threshold_at_fpr_sample(scores::Vector, y_true::Vector, fpr::Real, d::Real)
    @assert 0 <= d <= 1
    n = length(y_true)
    nn = floor(Int, d*n)
    sample_inds = StatsBase.sample(1:n, nn, replace=false)
    threshold_at_fpr(scores[sample_inds], y_true[sample_inds], fpr)
end

"""
    fpr_distribution(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int, d::Real=0.5)

Computes the distribution of false positive rates around given `fpr`.
"""
function fpr_distribution(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int, d::Real=0.5)
    thresholds = map(i->threshold_at_fpr_sample(scores, y_true, fpr, d), 1:nsamples)
    fprs = map(x->fpr_at_threshold(scores, y_true, x), thresholds)
end

"""
    localized_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int[; d::Real=0.5,
        normalize=false])

A localized AUC. Samples the distribution of false positive rate values around given `fpr`. Then
integrates the ROC curve only between found minimum and maximum value.

# Arguments
* `scores`: A numerical vector of sample scores, higher value indicates higher
probability of belonginig to the positive class
* `y_true`: True binary labels
* `fpr`: Value of FPR of interest
* `nsamples`: Number of FPR samples to be fitted
* `d`: A ratio of the size of the resampling set from which the distribution samples are drawn from
* `normalize`: Normalize the output?

"""
function localized_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5, 
    normalize=false)
    # first sample fprs and get parameters of the beta distribution
    fprs = fpr_distribution(scores, y_true, fpr, nsamples, d)
    roc = roccurve(scores, y_true)
    partial_auc(roc..., maximum(fprs), minimum(fprs), normalize=normalize)
end

"""
    estimate_beta_params(x::Vector)

Given a set of samples `x`, estimate the parameters of the one-dimensional Beta(α,β) distribution
as in `https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters`. 
"""
function estimate_beta_params(x::Vector)
    μ, σ2 = mean_and_var(x)
    if σ2 < μ*(1-μ)
        α = μ*(μ*(1-μ)/σ2 - 1)
        β = (1-μ)*(μ*(1-μ)/σ2 - 1)
    else
        α, β = NaN, NaN
    end
    return α, β
end

"""
    beta_logpdf(x::Real, α::Real, β::Real)

Log-pdf of Beta(α, β) distribution. Equals to log of x^(α-1)*(1-x)^(β-1)*Γ(α+β)/Γ(α)*Γ(β).
"""
beta_logpdf(x::Real, α::Real, β::Real) = 
    (α-1)*log(x) + (β-1)*log(1-x) + loggamma(α+β) - loggamma(α) - loggamma(β)

"""
    beta_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5)

Computes βAUC - an integral over ROC weighted by a Beta distribution of false positive rates
around `fpr`.

# Arguments
* `scores`: A numerical vector of sample scores, higher value indicates higher
probability of belonginig to the positive class
* `y_true`: True binary labels
* `fpr`: Value of FPR of interest
* `nsamples`: Number of FPR samples to be fitted
* `d`: A ratio of the size of the resampling set from which the distribution samples are drawn from

"""
function beta_auc(scores::Vector, y_true::Vector, fpr::Real, nsamples::Int; d::Real=0.5)
    # first sample fprs and get parameters of the beta distribution
    fprs = fpr_distribution(scores, y_true, fpr, nsamples, d)
    α, β = estimate_beta_params(fprs)

    # compute roc
    roc = roccurve(scores, y_true)
    
    # linearly interpolate it
    interp_len = max(1001, length(roc[1]))
    roci = linear_interpolation(roc..., n=interp_len)

    # weights are given by the beta pdf and are centered on the trapezoids
    dx = (roci[1][2] - roci[1][1])/2
    xw = roci[1][1:end-1] .+ dx
    w = exp.(beta_logpdf.(xw, α, β))
    wauroc = auc(roci..., w)
end

"""
    linear_interpolation(x::Vector,y::Vector;n=nothing,dx=nothing)

Linearly interpolate `x` and `y`.
"""
function linear_interpolation(x::Vector,y::Vector;n=nothing,dx=nothing)
    (n==nothing && dx==nothing) ? 
        error("Support one of the keyword arguments - number of steps `n` or step length `dx`") : nothing
    _x = (dx == nothing) ?
        collect(range(minimum(x), maximum(x), length=n)) : collect(range(minimum(x), maximum(x), step=dx))
    n = length(_x)
    _y = zeros(n)
    # now for an element in _x, compute the element in _y as a linear interpolation between the 
    # two elements in y
    for i in 1:n
        if i==1
            _y[i] = y[1]
        elseif i==n
            _y[i] = y[end]
        else
            ri = findfirst(_x[i].<x)
            li = findlast(_x[i].>=x)
            _y[i] = y[li] + (y[ri] - y[li])/(x[ri]-x[li])*(_x[i]-x[li])
        end
    end
    return _x, _y
end