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

function test_network_flow_planning()
    # Parameters
    optimizer = HiGHS.Optimizer
    n_edges = 4
    n_nodes = 4
    n_commodities = 2
    dist = Uniform(10, 90)

    #Generate Problem
    edges = Set{Tuple{Int,Int}}()
    perm = shuffle(collect(1:n_nodes))
    for i in 2:n_nodes
        u = perm[i]
        v = perm[rand(1:i-1)]
        push!(edges, (u,v))
    end

    # Add extra edges
    while length(edges) < n_edges
        u, v = rand(1:n_nodes), rand(1:n_nodes)
        if u != v
            push!(edges, (u,v))
        end
    end
    edges = collect(edges)

    commodities = Tuple{Int,Int}[]
    while length(commodities) < n_commodities
        orig, dest = rand(1:n_nodes), rand(1:n_nodes)
        if orig != dest
            push!(commodities, (orig, dest))
        end
    end

    cap_cost = rand(Uniform(10, 50), n_edges)

    # Init LDR
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    @variable(ldr, u[1:n_edges] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, f[1:n_edges, 1:n_commodities] .>= 0)

    dist_d = [dist for _ in 1:n_commodities]
    @variable(ldr, d[i = 1:n_commodities] in 
                LinearDecisionRules.ScalarUncertainty(dist_d[i]))

    dist_c = [dist for _ in 1:n_edges]
    @variable(ldr, c[i = 1:n_edges] in
                LinearDecisionRules.ScalarUncertainty(dist_c[i]))

    for (k, (orig, dest)) in enumerate(commodities)
        for v in 1:n_nodes
            outgoing = [e for (e,(i,j)) in enumerate(edges) if i == v]
            incoming = [e for (e,(i,j)) in enumerate(edges) if j == v]

            rhs = 0.0
            if v == orig
                rhs = d[k]
            elseif v == dest
                rhs = -d[k]
            end

            @constraint(ldr,
                sum(f[e, k] for e in outgoing) -
                sum(f[e, k] for e in incoming) == rhs
            )
        end
        incoming_dest = [e for (e,(i,j)) in enumerate(edges) if j == dest]
        @constraint(ldr,
            sum(f[e, k] for e in incoming_dest) >= d[k]
        )
    end

    for e in 1:n_edges
        @constraint(ldr, sum(f[e, k] for k in 1:n_commodities) <= u[e])
    end

    @objective(ldr, Min,
        sum(cap_cost[e] * u[e] for e in 1:n_edges) +
        sum(c[e] * f[e, k] for e in 1:n_edges, k in 1:n_commodities)
    )

    optimize!(ldr)

    pwldr = PiecewiseLDR.PWLDR(ldr)

    for i in 1:n_commodities
        PiecewiseLDR.set_breakpoint!(pwldr, d[i], 2)
    end

    for j in 1:n_edges
        PiecewiseLDR.set_breakpoint!(pwldr, c[j], 2)
    end

    optimize!(pwldr)
    obj_pwldr_not_opt = objective_value(pwldr)

    for i in 1:n_commodities
        LinearDecisionRules.set_attribute(
                d[i],
                LinearDecisionRules.BreakPoints(),
                2,)
    end

    for j in 1:n_edges
        LinearDecisionRules.set_attribute(
                c[j],
                LinearDecisionRules.BreakPoints(),
                2,)
    end
    optimize!(ldr)
    obj_ldr = objective_value(ldr)

    X_ldr = value.(ldr.primal_model[:X])
    X_pwldr = value.(pwldr.model[:X])

    C_ldr = ldr.ext[:_LDR_ABC].C
    C_pwldr = pwldr.model.ext[:C]
    @show C_ldr
    @show C_pwldr

    M_ldr = ldr.ext[:_LDR_M]
    M_pwldr = PiecewiseLDR._build_second_moment_matrix(pwldr.n_segments_vec, pwldr.PWVR_list)
    @show M_ldr
    @show M_pwldr

    @test isapprox(objective_value(ldr), objective_value(pwldr); atol=1e-6)

    PiecewiseLDR.local_search!(pwldr)
    optimize!(pwldr)
    obj_pwldr_opt = objective_value(pwldr)

    @test obj_ldr >= obj_pwldr_not_opt >= obj_pwldr_opt
end

#End Module
end

TestMain.runtests()