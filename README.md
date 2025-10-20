# PiecewiseLDR
Library to extend the LinearDecisionRules package to the piecewise ldr format

# Example
```julia
using Distributions
using JuMP
using HiGHS
using LinearDecisionRules

# It's not possible to install the package at the moment
include("src/PiecewiseLDR.jl")
using .PiecewiseLDR

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

# Build the piecewise model
pwldr = PiecewiseLDR.PWLDR(ldr)

# Set 2 breakpoints for the variable 
PiecewiseLDR.set_breakpoint!(pwldr, demand, 2)

# Optimize the location of breakpoints with local search model
PiecewiseLDR.local_search!(pwldr)

optimize!(pwldr)

@show objective_value(pwldr)
```