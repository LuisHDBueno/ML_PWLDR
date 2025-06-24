using Distributions
using JuMP
using Statistics
using Random
using LinearDecisionRules
using LinearAlgebra
using SparseArrays
using BlackBoxOptim

import JuMP: optimize!, objective_value, getindex

include("pwrv.jl")
include("canonical_transform.jl")
include("second_moment.jl")
include("segments/displace_segments.jl")
include("segments/segments_number.jl")

#displace_segments_models
include("segments/displace_segments_models/black_box.jl")
include("segments/displace_segments_models/local_search.jl")

function _build_problem(
    ABC,
    first_stage_index,
    PWVR_list::Vector{PWVR},
    n_segments_vec,
    optimizer
)

    model = Model(optimizer)
    set_silent(model)

    dim_X = size(ABC.Ae, 2)
    dim_ξ = Int(sum(n_segments_vec) + 1)

    η_min = ABC.lb

    @variable(model, X[1:dim_X, 1:dim_ξ])
    for i in first_stage_index
        @constraint(model, X[i, 2:end] .== 0)
    end

    # Equality constraints
    if size(ABC.Be, 1) > 0
        Be = _build_B(ABC.Be, η_min, n_segments_vec)
        @constraint(model, ABC.Ae * X .== Be)
    end

    W = _build_W(n_segments_vec, PWVR_list)
    h = _build_h(n_segments_vec)

    nW = size(W, 1)

    # Inequality contraints
    if size(ABC.Bu, 1) > 0
        Bu = _build_B(ABC.Bu, η_min, n_segments_vec)
        @variable(model, Su[1:size(Bu, 1), 1:dim_ξ] >= 0)
        @constraint(model, ABC.Au * X .+ Su .== Bu)
        @variable(model, ΛSu[1:size(Bu, 1),1:nW] >= 0)
        @constraint(model, ΛSu * W .== Su)
        @constraint(model, ΛSu * h .>= 0)
    end

    if size(ABC.Bl, 1) > 0
        Bl = _build_B(ABC.Bl, η_min, n_segments_vec)
        @variable(model, Sl[1:size(Bl, 1), 1:dim_ξ] >= 0)
        @constraint(model, ABC.Al * X .- Sl .== Bl)
        @variable(model, ΛSl[1:size(Bl, 1),1:nW] >= 0)
        @constraint(model, ΛSl * W .== Sl)
        @constraint(model, ΛSl * h .>= 0)
    end

    # Can only include rows where the bound is not +Inf
    idxs_xu = findall(x -> x != Inf, ABC.xu)
    if size(idxs_xu, 1) > 0
        @variable(model, Sxu[idxs_xu, 1:dim_ξ])
        @constraint(model, X[idxs_xu,1] .+ Sxu[idxs_xu,1] .== ABC.xu[idxs_xu])
        @constraint(model, X[idxs_xu,2:end] .+ Sxu[idxs_xu,2:end] .== 0)

        @variable(model, ΛSxu[idxs_xu,1:nW] >= 0)
        @constraint(model, ΛSxu.data * W .== Sxu.data)
        @constraint(model, ΛSxu.data * h .>= 0)
    end

    # Can only include rows where the bound is not -Inf
    idxs_xl = findall(x -> x != -Inf, ABC.xl)
    if size(idxs_xl, 1) > 0
        @variable(model, Sxl[idxs_xl, 1:dim_ξ])
        @constraint(model, X[idxs_xl,1] .- Sxl[idxs_xl,1] .== ABC.xl[idxs_xl])
        @constraint(model, X[idxs_xl,2:end] .- Sxl[idxs_xl,2:end] .== 0)

        @variable(model, ΛSxl[idxs_xl,1:nW] >= 0)
        @constraint(model, ΛSxl.data * W .== Sxl.data)
        @constraint(model, ΛSxl.data * h .>= 0)
    end

    C  = _build_C(ABC.C, η_min, n_segments_vec)

    model.ext[:C] = C

    M = _build_second_moment_matrix(n_segments_vec, PWVR_list)

    @objective(model, Min, LinearAlgebra.tr(C' * X * M))

    return model
end

struct PWLDR
    model::JuMP.Model
    PWVR_list::Vector{PWVR}
end

function  optimize!(model::PWLDR)
    JuMP.optimize!(model.model)
end

function objective_value(model::PWLDR)
    return objective_value(model.model)
end

function evaluate_sample(PWVR_list, X, C, samples)

    ξ = [1.0]
    for (pwvr, sp) in zip(PWVR_list, samples)
        η_vec = pwvr.η_vec
        ξ_ext = zeros(length(η_vec) - 1)
        value_cummulative = η_vec[1]
        for i in 1:(length(η_vec) - 1)
            diff = η_vec[i + 1] - η_vec[i]
            value_cummulative += diff
            if value_cummulative <= sp
                ξ_ext[i] = diff
            else
                ξ_ext[i] = sp - sum(ξ_ext) - η_vec[1]
                break
            end
        end
        append!(ξ, ξ_ext)
    end
    value_ret = ξ' * C' * X * ξ
    return value_ret
end

function getindex(model::PWLDR, indice::Symbol)
    return getindex(model.model, indice)
end

function PWLDR(ldr_model::LinearDecisionRules.LDRModel,
                optimizer;
                distribution_constructor::Type = Uniform,
                n_max_iter::Int = 50)
    ABC = ldr_model.ext[:_LDR_ABC]
    first_stage_index = ldr_model.ext[:_LDR_first_stage_indices]

    dim_ξ_ldr = size(ABC.Be, 2)

    n_segments_vec = zeros(dim_ξ_ldr - 1)
    dim_ξ = 1
    for i in 1:(dim_ξ_ldr - 1)
        n_segments = _segments_number()
        dim_ξ += n_segments
        n_segments_vec[i] = n_segments
    end

    η_min = ABC.lb
    η_max = ABC.ub

    #TODO: Change to optional displace optimizer, Black box optimizer as pattern 
    PWVR_list = _displace_segments(η_min, η_max, ABC, first_stage_index, n_segments_vec, optimizer, distribution_constructor)

    return PWLDR(_build_problem(ABC, first_stage_index, PWVR_list, n_segments_vec, optimizer), PWVR_list)
end