module EvalCurves
using LinearAlgebra
using SpecialFunctions
using StatsBase

export roccurve, auc, auc_at_p, tpr_at_fpr, prcurve, partial_auc

include("utils.jl")
include("np_score.jl")

end