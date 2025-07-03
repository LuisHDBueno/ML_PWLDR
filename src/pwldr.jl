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
include("segments/segments_number.jl")

struct PWLDR
    model::JuMP.Model
    PWVR_list::Vector{PWVR}
    n_segments_vec::Vector{Int}
    W_constraints::Dict{Symbol, Matrix{JuMP.ConstraintRef}}
end

function _build_init_pwvr_list(
    n_segments_vec::Vector{Int},
    η_min::Vector{Float64},
    η_max::Vector{Float64},
    distribution_constructor
    )
    pwvr_list = Vector{PWVR}()
    for i in 1:length(η_min)
        n = n_segments_vec[i]
        push!(pwvr_list,
                PWVR(distribution_constructor(η_min[i], η_max[i]),
                        η_min[i],
                        η_max[i],
                        fill(1/n, Int(n)))
                )
    end
    return pwvr_list
end

function _build_problem(
    ldr_model::LinearDecisionRules.LDRModel,
    n_segments_vec::Vector{Int},
    distribution_constructor,
    optimizer
)
    ABC = ldr_model.ext[:_LDR_ABC]
    first_stage_index = ldr_model.ext[:_LDR_first_stage_indices]

    η_min = ABC.lb
    η_max = ABC.ub
    pwvr_list = _build_init_pwvr_list(n_segments_vec, η_min, η_max,
                                         distribution_constructor)

    # Model Init
    model = Model(optimizer)
    set_silent(model)

    dim_X = size(ABC.Ae, 2)
    dim_ξ = Int(sum(n_segments_vec) + 1)

    @variable(model, X[1:dim_X, 1:dim_ξ])
    for i in first_stage_index
        @constraint(model, X[i, 2:end] .== 0)
    end

    # Equality constraints
    if size(ABC.Be, 1) > 0
        Be = _build_B(ABC.Be, η_min, n_segments_vec)
        @constraint(model, ABC.Ae * X .== Be)
    end

    #Reference each W dependent constraint                        
    W_constraints = Dict{Symbol, Matrix{JuMP.ConstraintRef}}()
    W = _build_W(n_segments_vec, pwvr_list)
    h = _build_h(n_segments_vec)

    nW = size(W, 1)

    # Inequality contraints
    if size(ABC.Bu, 1) > 0
        Bu = _build_B(ABC.Bu, η_min, n_segments_vec)
        @variable(model, Su[1:size(Bu, 1), 1:dim_ξ] >= 0)
        @constraint(model, ABC.Au * X .+ Su .== Bu)
        @variable(model, ΛSu[1:size(Bu, 1),1:nW] >= 0)
        
        #W dependence
        W_constraints[:upper_ineq] = @constraint(model, ΛSu * W .== Su)
        @constraint(model, ΛSu * h .>= 0)
    end

    if size(ABC.Bl, 1) > 0
        Bl = _build_B(ABC.Bl, η_min, n_segments_vec)
        @variable(model, Sl[1:size(Bl, 1), 1:dim_ξ] >= 0)
        @constraint(model, ABC.Al * X .- Sl .== Bl)
        @variable(model, ΛSl[1:size(Bl, 1),1:nW] >= 0)

        #W dependence
        W_constraints[:lower_ineq] =  @constraint(model, ΛSl * W .== Sl)
        @constraint(model, ΛSl * h .>= 0)
    end

    # Can only include rows where the bound is not +Inf
    idxs_xu = findall(x -> x != Inf, ABC.xu)
    if size(idxs_xu, 1) > 0
        @variable(model, Sxu[idxs_xu, 1:dim_ξ])
        @constraint(model, X[idxs_xu,1] .+ Sxu[idxs_xu,1] .== ABC.xu[idxs_xu])
        @constraint(model, X[idxs_xu,2:end] .+ Sxu[idxs_xu,2:end] .== 0)

        @variable(model, ΛSxu[idxs_xu,1:nW] >= 0)

        #W dependence
        W_constraints[:upper_x] = @constraint(model, ΛSxu.data * W .== Sxu.data)
        @constraint(model, ΛSxu.data * h .>= 0)
    end

    # Can only include rows where the bound is not -Inf
    idxs_xl = findall(x -> x != -Inf, ABC.xl)
    if size(idxs_xl, 1) > 0
        @variable(model, Sxl[idxs_xl, 1:dim_ξ])
        @constraint(model, X[idxs_xl,1] .- Sxl[idxs_xl,1] .== ABC.xl[idxs_xl])
        @constraint(model, X[idxs_xl,2:end] .- Sxl[idxs_xl,2:end] .== 0)

        @variable(model, ΛSxl[idxs_xl,1:nW] >= 0)
        
        #W dependence
        W_constraints[:lower_x] = @constraint(model, ΛSxl.data * W .== Sxl.data)
        @constraint(model, ΛSxl.data * h .>= 0)
    end

    C  = _build_C(ABC.C, η_min, n_segments_vec)

    model.ext[:C] = C

    M = _build_second_moment_matrix(n_segments_vec, pwvr_list)

    @objective(model, Min, LinearAlgebra.tr(C' * X * M))

    return PWLDR(model, pwvr_list, n_segments_vec, W_constraints)
end

function  optimize!(model::PWLDR)
    JuMP.optimize!(model.model)
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
    model = pwldr.model
    if haskey(pwldr.W_constraints, :upper_ineq)
        delete_matrix_constraint(model, pwldr.W_constraints[:upper_ineq])
        ΛSu = model[:ΛSu]
        Su = model[:Su]
        pwldr.W_constraints[:upper_ineq] = @constraint(model, ΛSu * W .== Su)
    end

    if haskey(pwldr.W_constraints, :lower_ineq)
        delete_matrix_constraint(model, pwldr.W_constraints[:lower_ineq])
        ΛSl = model[:ΛSl]
        Sl = model[:Sl]
        pwldr.W_constraints[:lower_ineq] =  @constraint(model, ΛSl * W .== Sl)
    end

    if haskey(pwldr.W_constraints, :upper_x)
        delete_matrix_constraint(model, pwldr.W_constraints[:upper_x])
        ΛSxu = model[:ΛSxu]
        Sxu = model[:Sxu]
        pwldr.W_constraints[:upper_x] = @constraint(model, ΛSxu.data * W .== Sxu.data)
    end

    if haskey(pwldr.W_constraints, :lower_x)
        delete_matrix_constraint(model, pwldr.W_constraints[:lower_x])
        ΛSxl = model[:ΛSxl]
        Sxl = model[:Sxl]
        pwldr.W_constraints[:lower_x] = @constraint(model, ΛSxl.data * W .== Sxl.data)
    end

    X = model[:X]
    C = model.ext[:C]
    
    M = _build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWVR_list)
    @objective(model, Min, LinearAlgebra.tr(C' * X * M))
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
                optimizer,
                distribution_constructor)

    #Get the ideal number of segments
    n_segments_vec = _segments_number(ldr_model)
    
    #Build the initial PWLDR problem
    pwldr_model = _build_problem(ldr_model, n_segments_vec,
                                    distribution_constructor, optimizer)

    return pwldr_model
end