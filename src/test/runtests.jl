module TestMain

using Distributions
using JuMP
using LinearDecisionRules
using HiGHS
using Random

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

function test_build_distribution_set()
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

    distributions = [
        Uniform(demand_min, demand_max),
        Uniform(demand_min, demand_max),
        Uniform(demand_min, demand_max)
    ]

    @variable(ldr, demand[i = 1:3] in LinearDecisionRules.ScalarUncertainty(distributions[i]), base_name="demand")

    @constraint(ldr, sell + ret <= buy)

    for i in 1:3
        @constraint(ldr, sell <= demand[i])
    end

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

function test_statistics()
    data = [10.0, 2.0, 38.0, 23.0, 38.0, 23.0, 21.0]
    v = PiecewiseLDR.get_statistics(data)

    @test 2.0 == v.min == v[1]
    @test 38.0 == v.max == v[2]
    @test isapprox(22.143, v.mean, atol=1e-3)
    @test v.mean == v[3]
    @test v.median == 23.00 == v[4]
    @test isapprox(12.298, v.std, atol=1e-3)
    @test v.std == v[5]
end

function test_set_opt_breakpoint_number()
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
    obj_pwldr_not_opt = objective_value(pwldr)

    PiecewiseLDR.set_opt_breakpoint_number!(pwldr, demand)
    optimize!(pwldr)
    obj_pwldr_opt = objective_value(pwldr)

    @test obj_pwldr_not_opt >= obj_pwldr_opt
end

function test_vector_representation()
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

    result = PiecewiseLDR.vector_representation(pwldr, demand)
    @test result isa Vector{Float64}
    @test size(result) == (15,)
end

function test_shipment_planing()
    # Parameters
    optimizer = HiGHS.Optimizer
    n_products = 1
    n_clients = 2
    prod_cost_1 = rand(Uniform(50, 100), n_products)
    prod_cost_2 = prod_cost_1 .+ rand(Uniform(99, 100), n_products)
    client_cost = rand(Uniform(25, 50), n_products, n_clients)
    dist = Uniform(10, 90)

    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    @variable(ldr, buy_1[1:n_products] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, buy_2[1:n_products] .>= 0)
    @variable(ldr, ship[1:n_products, 1:n_clients] .>= 0)

    distributions = [dist for _ in 1:n_clients]
    @variable(ldr, demand[i = 1:n_clients] in LinearDecisionRules.ScalarUncertainty(distributions[i]))

    for j in 1:n_clients
        @constraint(ldr, sum(ship[i, j] for i in 1:n_products) >= demand[j])
    end
    for i in 1:n_products
        @constraint(ldr, sum(ship[i, j] for j in 1:n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(ldr, Min,
                + sum(prod_cost_1 .* buy_1)
                + sum(prod_cost_2 .* buy_2)
                + sum(sum(client_cost .* ship)))
            
    optimize!(ldr)
    obj_ldr = objective_value(ldr)

    pwldr = PiecewiseLDR.PWLDR(ldr)

    for i in 1:n_clients
        PiecewiseLDR.set_breakpoint!(pwldr, demand[i], 2)
    end

    optimize!(pwldr)
    obj_pwldr_not_opt = objective_value(pwldr)

    for i in 1:n_clients
        LinearDecisionRules.set_attribute(
                demand[i],
                LinearDecisionRules.BreakPoints(),
                2,)
    end
    optimize!(ldr)

    @test isapprox(objective_value(ldr), objective_value(pwldr); atol=1e-6)

    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    obj_pwldr_opt = objective_value(pwldr)

    @test obj_ldr >= obj_pwldr_not_opt >= obj_pwldr_opt
end

#End Module
end

TestMain.runtests()