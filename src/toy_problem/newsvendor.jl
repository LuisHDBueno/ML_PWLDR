using Distributions
using Plots
using JSON
using JuMP
using HiGHS
using Statistics
using Random
using LinearDecisionRules
using LinearAlgebra
include("../pwldr.jl")

buy_cost = 10
return_value = 8
sell_value = 15

demand_max = 120
demand_min = 80

ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
set_silent(ldr)

@variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
@variable(ldr, sell >= 0)
@variable(ldr, ret >= 0)
@variable(ldr, demand in LinearDecisionRules.Uncertainty(
        distribution = Uniform(demand_min, demand_max)
    )
)

@constraint(ldr, sell + ret <= buy)
@constraint(ldr, sell <= demand)

@objective(ldr, Max,
    - buy_cost * buy
    + return_value * ret
    + sell_value * sell
)
optimize!(ldr)
@show objective_value(ldr)

model = PWLDR(ldr, HiGHS.Optimizer)
optimize!(model)
@show objective_value(model)
@show value.(model[:X])
@show model.PWVR_list[1].η_vec

