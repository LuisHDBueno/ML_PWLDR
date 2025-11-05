struct NetworkFlowPlanning
    n_nodes::Int
    n_edges::Int
    edges::Vector{Tuple{Int,Int}}
    n_commodities::Int
    commodities::Vector{Tuple{Int,Int}}
    cap_cost::Vector{Float64}
    demand_samples::Matrix{Float64}
    cost_samples::Matrix{Float64}
    probs::Vector{Float64}
    distribution::Distribution
    penalty::Float64
    optimizer
end

mutable struct NFPModel
    problem::NetworkFlowPlanning
    first_stage_decision
    model
    objective_value
    test_value
end

function NFPModelPWLDR(
    problem::NetworkFlowPlanning,
    pwldr::PiecewiseLDR.PWLDR
)
    optimize!(pwldr)
    obj_value = objective_value(pwldr)
    X = pwldr.model[:X]
    first_stage_index = sort(collect(pwldr.model.ext[:first_stage_index]))
    first_stage_decision = value.(X[first_stage_index,1])
    return NFPModel(problem, first_stage_decision, pwldr, obj_value, 0)
end

function generate_networkflow_problem(
    n_nodes::Int,
    n_edges::Int,
    n_commodities::Int,
    n_samples_train::Int,
    n_samples_test::Int,
    dist,
    optimizer;
    rng = Random.default_rng()
)

    #Build Spanning Tree
    edges = Set{Tuple{Int,Int}}()
    perm = shuffle(collect(1:n_nodes), rng)
    for i in 2:n_nodes
        u = perm[i]
        v = perm[rand(rng, 1:i-1)]
        push!(edges, (u,v))
    end

    # Add extra edges
    while length(edges) < n_edges
        u, v = rand(rng, 1:n_nodes), rand(rng, 1:n_nodes)
        if u != v
            push!(edges, (u,v))
        end
    end
    edges = collect(edges)

    commodities = Tuple{Int,Int}[]
    while length(commodities) < n_commodities
        orig, dest = rand(rng, 1:n_nodes), rand(rng, 1:n_nodes)
        if orig != dest
            push!(commodities, (orig, dest))
        end
    end

    cap_cost = rand(rng, Uniform(10, 50), n_edges)

    probs = fill(1.0/n_samples_train, n_samples_train)

    samples_demand_train = rand(rng, dist, n_commodities, n_samples_train)
    samples_demand_test  = rand(rng, dist, n_commodities, n_samples_test)

    samples_cost_train = rand(rng, dist, n_edges, n_samples_train)
    samples_cost_test  = rand(rng, dist, n_edges, n_samples_test)

    penalty = 1e6

    problem = NetworkFlowPlanning(
        n_nodes,
        n_edges,
        edges,
        n_commodities,
        commodities,
        cap_cost,
        samples_demand_train,
        samples_cost_train,
        probs,
        dist,
        penalty,
        optimizer
    )

    return problem, samples_demand_test, samples_cost_test
end

function nfp_second_stage(
    sp_model::NFPModel,
    demand_samples_test,
    cost_samples_test
)
    problem = sp_model.problem
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    u = sp_model.first_stage_decision
    @variable(model, f[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(model, short[1:problem.n_commodities] .>= 0)
    @variable(model, relax_pos[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(model, relax_neg[1:problem.n_edges, 1:problem.n_commodities] .>= 0)

    penalty = problem.penalty

    demand_constraints = Matrix{ConstraintRef}(undef, problem.n_commodities, problem.n_nodes)
    min_demand_constraints = Vector{ConstraintRef}(undef, problem.n_commodities)
    for (k, (orig, dest)) in enumerate(problem.commodities)
        for v in 1:problem.n_nodes
            outgoing = [e for (e,(i,j)) in enumerate(problem.edges) if i == v]
            incoming = [e for (e,(i,j)) in enumerate(problem.edges) if j == v]

            demand_constraints[k, v] = @constraint(model,
                                            sum(f[e, k] for e in outgoing) -
                                            sum(f[e, k] for e in incoming) +
                                            relax_pos[v, k] - relax_neg[v, k] == 0.0
                                        )
        end
        incoming_dest = [e for (e,(i,j)) in enumerate(problem.edges) if j == dest]
        min_demand_constraints[k] = @constraint(model, sum(f[e, k] for e in incoming_dest) + short[k] >= 0.0)
    end

    for e in 1:problem.n_edges
        @constraint(model, sum(f[e, k] for k in 1:problem.n_commodities) <= u[e])
    end

    @objective(model, Min, 0.0)

    total = 0
    S = size(demand_samples_test, 2)
    for i in 1:S
        d = demand_samples_test[:, i]
        c = cost_samples_test[:, i]
        for (k, (orig, dest)) in enumerate(problem.commodities)
            for v in 1:problem.n_nodes
                rhs = 0.0
                if v == orig
                    rhs = d[k]
                elseif v == dest
                    rhs = -d[k]
                end
                set_normalized_rhs(demand_constraints[k, v], rhs)
            end
            set_normalized_rhs(min_demand_constraints[k], d[k])
        end
            
        expr = sum(problem.cap_cost[e] * u[e] for e in 1:problem.n_edges) +
                sum(c[e] * f[e, k] for e in 1:problem.n_edges, k in 1:problem.n_commodities) +
                sum(penalty * short[k] for k in 1:problem.n_commodities) +
                sum(penalty * (relax_pos[v,k] + relax_neg[v,k])
                    for v in 1:problem.n_nodes, k in 1:problem.n_commodities)

        set_objective_function(model, expr)

        optimize!(model)
        total += objective_value(model)
    end

    return total/S
end

function nfp_ldr(problem::NetworkFlowPlanning, dist)
    ldr = LinearDecisionRules.LDRModel(problem.optimizer)
    set_silent(ldr)

    @variable(ldr, u[1:problem.n_edges] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, f[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(ldr, short[1:problem.n_commodities] .>= 0)
    @variable(ldr, relax_pos[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(ldr, relax_neg[1:problem.n_edges, 1:problem.n_commodities] .>= 0)

    penalty = problem.penalty

    @variable(ldr, d[i = 1:problem.n_commodities] in 
                LinearDecisionRules.ScalarUncertainty(dist))

    @variable(ldr, c[i = 1:problem.n_edges] in
                LinearDecisionRules.ScalarUncertainty(dist))

    for (k, (orig, dest)) in enumerate(problem.commodities)
        for v in 1:problem.n_nodes
            outgoing = [e for (e,(i,j)) in enumerate(problem.edges) if i == v]
            incoming = [e for (e,(i,j)) in enumerate(problem.edges) if j == v]

            rhs = 0.0
            if v == orig
                rhs = d[k]
            elseif v == dest
                rhs = -d[k]
            end

            @constraint(ldr,
                sum(f[e, k] for e in outgoing) -
                sum(f[e, k] for e in incoming) +
                relax_pos[v, k] - relax_neg[v, k]== rhs
            )
        end
        incoming_dest = [e for (e,(i,j)) in enumerate(problem.edges) if j == dest]
        @constraint(ldr,
            sum(f[e, k] for e in incoming_dest) >= d[k] - short[k]
        )
    end

    for e in 1:problem.n_edges
        @constraint(ldr, sum(f[e, k] for k in 1:problem.n_commodities) <= u[e])
    end

    @objective(ldr, Min,
        sum(problem.cap_cost[e] * u[e] for e in 1:problem.n_edges) +
        sum(c[e] * f[e, k] for e in 1:problem.n_edges, k in 1:problem.n_commodities) +
        sum(penalty * short[k] for k in 1:problem.n_commodities) +
        sum(penalty * (relax_pos[v,k] + relax_neg[v,k])
            for v in 1:problem.n_nodes, k in 1:problem.n_commodities)
    )

    optimize!(ldr)

    X = ldr.primal_model[:X]
    first_stage_index = sort(collect(ldr.ext[:_LDR_first_stage_indices]))
    first_stage_decision = value.(X[first_stage_index,1])
    obj_value = objective_value(ldr)

    return NFPModel(problem, first_stage_decision, ldr, obj_value, 0)
end

function nfp_standard_form(problem::NetworkFlowPlanning)
    S = size(problem.demand_samples, 2)

    model = Model(problem.optimizer)
    set_silent(model)

    # 1 Stage
    @variable(model, u[1:problem.n_edges] .>= 0)

    # 2 Stage
    @variable(model, f[1:problem.n_edges, 1:problem.n_commodities, 1:S] .>= 0)
    @variable(model, short[1:problem.n_commodities, 1:S] .>= 0)
    @variable(model, relax_pos[1:problem.n_edges, 1:problem.n_commodities, 1:S] .>= 0)
    @variable(model, relax_neg[1:problem.n_edges, 1:problem.n_commodities, 1:S] .>= 0)

    penalty = problem.penalty

    for s in 1:S
        d = problem.demand_samples[:, s]
        for (k, (orig, dest)) in enumerate(problem.commodities)
            for v in 1:problem.n_nodes
                outgoing = [e for (e,(i,j)) in enumerate(problem.edges) if i == v]
                incoming = [e for (e,(i,j)) in enumerate(problem.edges) if j == v]

                rhs = 0.0
                if v == orig
                    rhs = d[k]
                elseif v == dest
                    rhs = -d[k]
                end

                @constraint(model,
                    sum(f[e, k, s] for e in outgoing) -
                    sum(f[e, k, s] for e in incoming) +
                    relax_pos[v,k,s] - relax_neg[v,k,s] == rhs
                )
            end
            incoming_dest = [e for (e,(i,j)) in enumerate(problem.edges) if j == dest]
            @constraint(model, sum(f[e, k, s] for e in incoming_dest) >= d[k] - short[k, s])
        end
        for e in 1:problem.n_edges
            @constraint(model, sum(f[e, k, s] for k in 1:problem.n_commodities) <= u[e])
        end
    end

    expr_first = sum(problem.cap_cost[e] * u[e] for e in 1:problem.n_edges)

    expr_second = 0.0
    for s in 1:S
        p = problem.probs[s]
        c_s = problem.cost_samples[:, s]
        expr_second += p * sum(c_s[e] * f[e,k,s] for e in 1:problem.n_edges, k in 1:problem.n_commodities)
    end

    expr_penalty =  sum(problem.probs[s] * penalty * short[k,s]
                        for s in 1:S, k in 1:problem.n_commodities) +
                    sum(problem.probs[s] * penalty * (relax_pos[v,k,s] + relax_neg[v,k,s])
                        for s in 1:S, v in 1:problem.n_nodes, k in 1:problem.n_commodities)

    @objective(model, Min, expr_first + expr_second + expr_penalty)

    optimize!(model)
    u_sol = value.(u)
    obj_val = objective_value(model)

    return NFPModel(problem, u_sol, model, obj_val, 0)
end

function nfp_eev(problem::NetworkFlowPlanning)
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, u[1:problem.n_edges] .>= 0)
    @variable(model, f[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(model, short[1:problem.n_commodities] .>= 0)
    @variable(model, relax_pos[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(model, relax_neg[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    
    penalty = problem.penalty

    for (k, (orig, dest)) in enumerate(problem.commodities)
        for v in 1:problem.n_nodes
            outgoing = [e for (e,(i,j)) in enumerate(problem.edges) if i == v]
            incoming = [e for (e,(i,j)) in enumerate(problem.edges) if j == v]

            rhs = 0.0
            if v == orig
                rhs = Distributions.mean(problem.distribution)
            elseif v == dest
                rhs = -Distributions.mean(problem.distribution)
            end

            @constraint(model,
                sum(f[e, k] for e in outgoing) -
                sum(f[e, k] for e in incoming) +
                relax_pos[v, k] - relax_neg[v, k] == rhs
            )
        end
        incoming_dest = [e for (e,(i,j)) in enumerate(problem.edges) if j == dest]
        @constraint(model, sum(f[e, k] for e in incoming_dest) >= Distributions.mean(problem.distribution) - short[k])
    end

    for e in 1:problem.n_edges
        @constraint(model, sum(f[e, k] for k in 1:problem.n_commodities) <= u[e])
    end

    @objective(model, Min,
        sum(problem.cap_cost[e] * u[e] for e in 1:problem.n_edges) +
        sum(Distributions.mean(problem.distribution) * f[e, k]
            for e in 1:problem.n_edges, k in 1:problem.n_commodities) +
        sum(penalty * short[k] for k in 1:problem.n_commodities) +
        sum(penalty * (relax_pos[v,k] + relax_neg[v,k])
            for v in 1:problem.n_nodes, k in 1:problem.n_commodities)
    )

    optimize!(model)

    first_stage_decision = value.(u)
    obj_value = objective_value(model)

    return NFPModel(problem, first_stage_decision, model, obj_value, 0)
end

function nfp_ws(
    problem::NetworkFlowPlanning,
    demand_samples_test,
    cost_samples_test
)
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, u[1:problem.n_edges] .>= 0)
    @variable(model, f[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(model, short[1:problem.n_commodities] .>= 0)
    @variable(model, relax_pos[1:problem.n_edges, 1:problem.n_commodities] .>= 0)
    @variable(model, relax_neg[1:problem.n_edges, 1:problem.n_commodities] .>= 0)

    penalty = problem.penalty

    demand_constraints = Matrix{ConstraintRef}(undef, problem.n_commodities, problem.n_nodes)
    min_demand_constraints = Vector{ConstraintRef}(undef, problem.n_commodities)
    for (k, (orig, dest)) in enumerate(problem.commodities)
        for v in 1:problem.n_nodes
            outgoing = [e for (e,(i,j)) in enumerate(problem.edges) if i == v]
            incoming = [e for (e,(i,j)) in enumerate(problem.edges) if j == v]

            demand_constraints[k, v] = @constraint(model,
                                            sum(f[e, k] for e in outgoing) -
                                            sum(f[e, k] for e in incoming) +
                                            relax_pos[v, k] - relax_neg[v, k] == 0.0
                                        )
        end
        incoming_dest = [e for (e,(i,j)) in enumerate(problem.edges) if j == dest]
        min_demand_constraints[k] = @constraint(model, sum(f[e, k] for e in incoming_dest) + short[k] >= 0.0)
    end

    for e in 1:problem.n_edges
        @constraint(model, sum(f[e, k] for k in 1:problem.n_commodities) <= u[e])
    end

    total = 0
    S = size(demand_samples_test, 2)
    for i in 1:S
        d = demand_samples_test[:, i]
        c = cost_samples_test[:, i]
        for (k, (orig, dest)) in enumerate(problem.commodities)
            for v in 1:problem.n_nodes
                rhs = 0.0
                if v == orig
                    rhs = d[k]
                elseif v == dest
                    rhs = -d[k]
                end
                set_normalized_rhs(demand_constraints[k, v], rhs)
            end
        end
        
        @objective(model, Min,
            sum(problem.cap_cost[e] * u[e] for e in 1:problem.n_edges) +
            sum(c[e] * f[e, k] for e in 1:problem.n_edges, k in 1:problem.n_commodities) +
            sum(penalty * short[k] for k in 1:problem.n_commodities) +
            sum(penalty * (relax_pos[v,k] + relax_neg[v,k])
                for v in 1:problem.n_nodes, k in 1:problem.n_commodities)
        )

        optimize!(model)
        total += objective_value(model)
    end

    return total/S
end
 