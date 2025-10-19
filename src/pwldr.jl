module PiecewiseLDR

using Distributions
using Expectations
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
include("segments/segments_number.jl")

mutable struct PWLDR
    model::JuMP.Model
    ldr_model::LinearDecisionRules.LDRModel
    PWVR_list::Vector{PWVR}
    n_segments_vec::Vector{Int}
    W_constraints::Dict{Symbol, Matrix{JuMP.ConstraintRef}}
    h_constraints::Dict{Symbol, Vector{JuMP.ConstraintRef}}
    reset_model::Bool
    uncertainty_to_distribution
end

function flatten_distributions_in_order(
    ldr_model::LinearDecisionRules.LDRModel
)
    scalar_distributions = ldr_model.cache_model.scalar_distributions
    vector_distributions = ldr_model.cache_model.vector_distributions
    uncertainty_to_distribution = ldr_model.cache_model.uncertainty_to_distribution

    variables = all_variables(ldr_model.cache_model.model)
    all_distributions = Vector{Distribution}()

    scalar_idx = 1
    vector_idx = 1
    temp = []

    for var in variables
        if !(var in keys(uncertainty_to_distribution))
            continue
        end
        dist_idx, inner_idx = uncertainty_to_distribution[var]

        if inner_idx == 0
            push!(temp, (dist_idx, scalar_distributions[scalar_idx]))
            scalar_idx += 1
        else
            push!(temp, (dist_idx, vector_distributions[vector_idx].v[inner_idx]))
            if inner_idx == length(vector_distributions[vector_idx].v)
                vector_idx += 1
            end
        end
    end

    # Agora ordena pelo dist_idx
    sorted_temp = sort(temp, by = x -> x[1])

    # Extrai apenas as distribuições e adiciona na ordem correta
    for (_, dist) in sorted_temp
        push!(all_distributions, dist)
    end

    return all_distributions
end

function _build_init_pwvr_list(
    n_segments_vec::Vector{Int},
    η_min::Vector{Float64},
    η_max::Vector{Float64},
    distribution_vec::Vector{Distribution}
    )
    pwvr_list = Vector{PWVR}()
    for i in 1:length(η_min)
        n = n_segments_vec[i]
        push!(pwvr_list,
                PWVR(truncated(distribution_vec[i], η_min[i], η_max[i]),
                        η_min[i],
                        η_max[i],
                        fill(1/n, Int(n)))
                )
    end
    return pwvr_list
end

function _build_problem(
    pwldr::PWLDR
)
    ldr_model = pwldr.ldr_model
    n_segments_vec = pwldr.n_segments_vec
    ABC = ldr_model.ext[:_LDR_ABC]
    first_stage_index = ldr_model.ext[:_LDR_first_stage_indices]

    η_min = ABC.lb
    η_max = ABC.ub

    distribution_vec = flatten_distributions_in_order(ldr_model)
    pwvr_list = _build_init_pwvr_list(n_segments_vec, η_min, η_max, distribution_vec)

    # Model Init
    model = pwldr.model
    set_silent(model)
    model.ext[:first_stage_index] = first_stage_index

    dim_X = size(ABC.Ae, 2)
    dim_ξ = Int(sum(n_segments_vec) + 1)

    @expression(model, X[1:dim_X, 1:dim_ξ], AffExpr(0.0))
    for i in 1:dim_X
        if i in first_stage_index
            X[i, 1] = @variable(model, base_name = "X[$i,1]")
        else
            for j in 1:dim_ξ
                X[i, j] = @variable(model, base_name = "X[$i,$j]")
            end
        end
    end

    #Reference each W dependent constraint                        
    W_constraints = Dict{Symbol, Matrix{JuMP.ConstraintRef}}()
    h_constraints = Dict{Symbol, Vector{JuMP.ConstraintRef}}()
    W = _build_W(n_segments_vec, pwvr_list)
    h = _build_h(pwvr_list)
    nW = size(W, 1)

    # Inequality contraints
    if size(ABC.Bu, 1) > 0
        Bu = _build_B(ABC.Bu, n_segments_vec)
        @variable(model, Su[1:size(Bu, 1), 1:dim_ξ])
        @constraint(model, ABC.Au * X .+ Su .== Bu)
        @variable(model, ΛSu[1:size(Bu, 1),1:nW] >= 0)

        #W, h dependence
        W_constraints[:upper_ineq] = @constraint(model, ΛSu * W .== Su)
        h_constraints[:upper_ineq] = @constraint(model, ΛSu * h .>= 0)
    end

    if size(ABC.Bl, 1) > 0
        Bl = _build_B(ABC.Bl, n_segments_vec)
        @variable(model, Sl[1:size(Bl, 1), 1:dim_ξ])
        @constraint(model, ABC.Al * X .- Sl .== Bl)
        @variable(model, ΛSl[1:size(Bl, 1),1:nW] >= 0)

        #W, h dependence
        W_constraints[:lower_ineq] = @constraint(model, ΛSl * W .== Sl)
        h_constraints[:lower_ineq] = @constraint(model, ΛSl * h .>= 0)
    end

    # Can only include rows where the bound is not +Inf
    idxs_xu = findall(x -> x != Inf, ABC.xu)
    if size(idxs_xu, 1) > 0
        @variable(model, Sxu[idxs_xu, 1:dim_ξ])
        @constraint(model, X[idxs_xu,1] .+ Sxu[idxs_xu,1] .== ABC.xu[idxs_xu])
        @constraint(model, X[idxs_xu,2:end] .+ Sxu[idxs_xu,2:end] .== 0)

        @variable(model, ΛSxu[idxs_xu,1:nW] .>= 0)

        #W, h dependence
        W_constraints[:upper_x] = @constraint(model, ΛSxu.data * W .== Sxu.data)
        h_constraints[:upper_x] = @constraint(model, ΛSxu.data * h .>= 0)
    end

    # Can only include rows where the bound is not -Inf
    idxs_xl = findall(x -> x != -Inf, ABC.xl)
    if size(idxs_xl, 1) > 0
        @variable(model, Sxl[idxs_xl, 1:dim_ξ])
        @constraint(model, X[idxs_xl,1] .- Sxl[idxs_xl,1] .== ABC.xl[idxs_xl])
        @constraint(model, X[idxs_xl,2:end] .- Sxl[idxs_xl,2:end] .== 0)

        @variable(model, ΛSxl[idxs_xl,1:nW] .>= 0)
        
        #W, h dependence
        W_constraints[:lower_x] = @constraint(model, ΛSxl.data * W .== Sxl.data)
        h_constraints[:lower_x] = @constraint(model, ΛSxl.data * h .>= 0)
    end

    C  = _build_C(ABC.C, n_segments_vec)

    model.ext[:C] = C
    M = _build_second_moment_matrix(n_segments_vec, pwvr_list)

    @expression(model, obj, LinearAlgebra.tr(C' * X * M))

    if ldr_model.ext[:_LDR_sense] == MOI.MIN_SENSE
        @objective(model, Min, obj)
    else
        @objective(model, Max, obj)
    end
    model.ext[:sense] = ldr_model.ext[:_LDR_sense]

    # Fill pwldr struct
    pwldr.PWVR_list = pwvr_list
    pwldr.W_constraints = W_constraints
    pwldr.h_constraints = h_constraints
    pwldr.reset_model = false
end

function optimize!(model::PWLDR)
    if model.reset_model
        _build_problem(model)
        JuMP.optimize!(model.model)
    else
        JuMP.optimize!(model.model)
    end
end

function objective_value(model::PWLDR)
    return objective_value(model.model)
end

function update_breakpoints!(pwldr::PWLDR, weight_vec::Vector{Vector{Float64}})
    function delete_matrix_constraint(model, matrix_constraint)
        for constr in matrix_constraint
            delete(model, constr)
        end
    end

    for i in 1:length(pwldr.PWVR_list)
        update_breakpoints!(pwldr.PWVR_list[i], weight_vec[i])
    end

    W = _build_W(pwldr.n_segments_vec, pwldr.PWVR_list)
    h = _build_h(pwldr.PWVR_list)
    model = pwldr.model
    if haskey(pwldr.W_constraints, :upper_ineq)
        delete_matrix_constraint(model, pwldr.W_constraints[:upper_ineq])
        delete(model, pwldr.h_constraints[:upper_ineq])
        ΛSu = model[:ΛSu]
        Su = model[:Su]
        pwldr.W_constraints[:upper_ineq] = @constraint(model, ΛSu * W .== Su)
        pwldr.h_constraints[:upper_ineq] = @constraint(model, ΛSu * h .>= 0)
    end

    if haskey(pwldr.W_constraints, :lower_ineq)
        delete_matrix_constraint(model, pwldr.W_constraints[:lower_ineq])
        delete(model, pwldr.h_constraints[:lower_ineq])
        ΛSl = model[:ΛSl]
        Sl = model[:Sl]
        pwldr.W_constraints[:lower_ineq] = @constraint(model, ΛSl * W .== Sl)
        pwldr.h_constraints[:lower_ineq] = @constraint(model, ΛSl * h .>= 0)
    end

    if haskey(pwldr.W_constraints, :upper_x)
        delete_matrix_constraint(model, pwldr.W_constraints[:upper_x])
        delete(model, pwldr.h_constraints[:upper_x])
        ΛSxu = model[:ΛSxu]
        Sxu = model[:Sxu]
        pwldr.W_constraints[:upper_x] = @constraint(model, ΛSxu.data * W .== Sxu.data)
        pwldr.h_constraints[:upper_x] = @constraint(model, ΛSxu.data * h .>= 0)
    end

    if haskey(pwldr.W_constraints, :lower_x)
        delete_matrix_constraint(model, pwldr.W_constraints[:lower_x])
        delete(model, pwldr.h_constraints[:lower_x])
        ΛSxl = model[:ΛSxl]
        Sxl = model[:Sxl]
        pwldr.W_constraints[:lower_x] = @constraint(model, ΛSxl.data * W .== Sxl.data)
        pwldr.h_constraints[:lower_x] = @constraint(model, ΛSxl.data * h .>= 0)
    end

    X = model[:X]
    C = model.ext[:C]
    
    M = _build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWVR_list)

    if model.ext[:sense] == MOI.MIN_SENSE
        @objective(model, Min, LinearAlgebra.tr(C' * X * M))
    else
        @objective(model, Max, LinearAlgebra.tr(C' * X * M))
    end
end

function evaluate_sample(PWVR_list, X, C, samples)
    #Change evaluate to correct sample order
    ξ = [1.0]
    for (pwvr, sp) in zip(PWVR_list, samples)
        ξ_ext = sample_vector(pwvr, sp)
        append!(ξ, ξ_ext)
    end
    value_ret = (C * ξ)' * (X * ξ)
    return value_ret
end

function getindex(model::PWLDR, indice::Symbol)
    return getindex(model.model, indice)
end

function PWLDR(ldr_model::LinearDecisionRules.LDRModel)
    
    n_segments_vec = _segments_number(ldr_model)

    #Build empty model
    empty_model = PWLDR(JuMP.Model(ldr_model.solver),
                        ldr_model,
                        PWVR[],
                        n_segments_vec,
                        Dict{Symbol, Matrix{JuMP.ConstraintRef}}(),
                        Dict{Symbol, Vector{JuMP.ConstraintRef}}(),
                        true,
                        ldr_model.cache_model.uncertainty_to_distribution)

    return empty_model
end

end #End module
