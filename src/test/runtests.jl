module TestMain

include("../pwldr.jl")
using .PiecewiseLDR

using LinearDecisionRules
using JuMP
using Distributions

using Test
using HiGHS

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_build_ldr()
    optimizer = HiGHS.Optimizer
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

    pwldr = PiecewiseLDR.PWLDR(ldr)

    optimize!(pwldr)

    @test objective_value(ldr) == objective_value(pwldr)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])

    @test isapprox(X_ldr, X_pwldr; atol=1e-6)

end

function test_build_vector_distribution()
    optimizer = HiGHS.Optimizer
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
    @variable(ldr, demand_before_vec in LinearDecisionRules.Uncertainty(
            distribution = Uniform(demand_min, demand_max)
        )
    )
    @variable(ldr, demand_vec[1:2] in LinearDecisionRules.Uncertainty(
            distribution = product_distribution([
                                        Uniform(demand_min, demand_max),
                                        Uniform(demand_min, demand_max)
                                    ])
        )
    )
    @variable(ldr, demand_after_vec in LinearDecisionRules.Uncertainty(
            distribution = Uniform(demand_min, demand_max)
        )
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell .<= demand_vec)
    @constraint(ldr, sell <= demand_before_vec)
    @constraint(ldr, sell <= demand_after_vec)

    @objective(ldr, Max,
        - buy_cost * buy
        + return_value * ret
        + sell_value * sell
    )
    optimize!(ldr)

    pwldr = PiecewiseLDR.PWLDR(ldr)

    optimize!(pwldr)

    @test objective_value(ldr) == objective_value(pwldr)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])

    @test isapprox(X_ldr, X_pwldr; atol=1e-6)
end

function test_build_pwldr()
    optimizer = HiGHS.Optimizer
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

    n_breakpoints = 2
    pwldr = PiecewiseLDR.PWLDR(ldr)
    PiecewiseLDR.set_breakpoint!(pwldr, demand, n_breakpoints)
    optimize!(pwldr)

    @test objective_value(ldr) <= objective_value(pwldr)
    
    LinearDecisionRules.set_attribute(
            demand,
            LinearDecisionRules.BreakPoints(),
            n_breakpoints,)
    optimize!(ldr)
    
    @test objective_value(ldr) == objective_value(pwldr)
    
    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])
    @test isapprox(X_ldr, X_pwldr; atol=1e-6)

    C_ldr = ldr.ext[:_LDR_ABC].C
    C_pwldr = pwldr.model.ext[:C]
    @test isapprox(C_ldr, C_pwldr; atol=1e-6)

    M_ldr = ldr.ext[:_LDR_M]
    M_pwldr = PiecewiseLDR._build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWVR_list)
    @test isapprox(M_ldr, M_pwldr; atol=1e-6)

end

#End Module
end

TestMain.runtests()