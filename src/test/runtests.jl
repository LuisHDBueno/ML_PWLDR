module TestMain

using Distributions
using JuMP
using LinearDecisionRules
using HiGHS

include("../PiecewiseLDR.jl")
using .PiecewiseLDR

using Test

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

function test_update_breakpoints()

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

    # update breakpoints
    η_vec = [80.0, 90.0, 110.0, 120.0]
    PiecewiseLDR.update_breakpoints!(pwldr, [[1.0,2.0,1.0]])

    LinearDecisionRules.set_attribute(
        demand,
        LinearDecisionRules.BreakPoints(),
        η_vec[2:end-1],)

    optimize!(pwldr)
    optimize!(ldr)

    @test isapprox(objective_value(ldr), objective_value(pwldr);atol=1e-6)
    
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

function test_black_box()
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
    before_opt_displace = objective_value(pwldr)

    PiecewiseLDR.black_box!(pwldr)
    optimize!(pwldr)
    after_opt_displace = objective_value(pwldr)

    @test before_opt_displace <= after_opt_displace
    
end

function test_local_search()
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
    before_opt_displace = objective_value(pwldr)

    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    after_opt_displace = objective_value(pwldr)

    @test before_opt_displace <= after_opt_displace
end

#End Module
end

TestMain.runtests()