struct ShipmentPlanning
    n_products::Int
    n_clients::Int
    prod_cost_1::Vector{Float64}
    prod_cost_2::Vector{Float64}
    client_cost::Matrix{Float64}
    samples_train
    probs
    distribution
    optimizer
end

mutable struct SPModel
    problem::ShipmentPlanning
    first_stage_decision
    model
    objective_value
    test_value
end

function SPModelPWLDR(
    problem::ShipmentPlanning,
    pwldr::PiecewiseLDR.PWLDR
)
    optimize!(pwldr)
    obj_value = objective_value(pwldr)
    X = pwldr.model[:X]
    first_stage_index = sort(collect(pwldr.model.ext[:first_stage_index]))
    first_stage_decision = value.(X[first_stage_index,1])
    return SPModel(problem, first_stage_decision, pwldr, obj_value, 0)
end

function sp_second_stage(
    sp_model::SPModel,
    samples_test
)

    problem = sp_model.problem
    # Solve with fixed first stage
    fixed_cost = sum(problem.prod_cost_1 .* sp_model.first_stage_decision)

    # Create model
    recourse_model = JuMP.Model(problem.optimizer)
    set_silent(recourse_model)

    @variable(recourse_model, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)
    @variable(recourse_model, buy_2[1:problem.n_products] .>= 0)

    for i in 1:problem.n_products
        @constraint(recourse_model, sum(ship[i, j] for j in 1:problem.n_clients) <= sp_model.first_stage_decision[i] + buy_2[i])
    end

    demand_constraints = Vector{ConstraintRef}(undef, problem.n_clients)
    for j in 1:problem.n_clients
        demand_constraints[j] = @constraint(recourse_model, sum(ship[i, j] for i in 1:problem.n_products) >= 0.0)
    end

    @objective(recourse_model, Min,
            sum(problem.prod_cost_2 .* buy_2) +
            sum(sum(problem.client_cost .* ship))
    )

    total_cost = 0.0
    for sample in samples_test
        for j in 1:problem.n_clients
            set_normalized_rhs(demand_constraints[j], sample[j])
        end
        optimize!(recourse_model)

        scenario_cost = fixed_cost + objective_value(recourse_model)
        total_cost += scenario_cost
    end

    return total_cost / length(samples_test)
end

function sp_ldr(
    problem::ShipmentPlanning,
    dist
)
    ldr = LinearDecisionRules.LDRModel(problem.optimizer)
    set_silent(ldr)

    @variable(ldr, buy_1[1:problem.n_products] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, buy_2[1:problem.n_products] .>= 0)
    @variable(ldr, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)

    @variable(ldr, demand[1:problem.n_clients] in LinearDecisionRules.Uncertainty(;
                                    distribution = product_distribution([
                                        dist for _ in 1:problem.n_clients
                                    ]),
                                    )
                )

    for j in 1:problem.n_clients
        @constraint(ldr, sum(ship[i, j] for i in 1:problem.n_products) >= demand[j])
    end
    for i in 1:problem.n_products
        @constraint(ldr, sum(ship[i, j] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(ldr, Min,
                + sum(problem.prod_cost_1 .* buy_1)
                + sum(problem.prod_cost_2 .* buy_2)
                + sum(sum(problem.client_cost .* ship)))
            
    optimize!(ldr)

    X = ldr.primal_model[:X]
    first_stage_index = sort(collect(ldr.ext[:_LDR_first_stage_indices]))
    first_stage_decision = value.(X[first_stage_index,1])
    obj_value = objective_value(ldr)

    return SPModel(problem, first_stage_decision, ldr, obj_value, 0)
end

function sp_standard_form(
    problem::ShipmentPlanning
)
    S = length(problem.samples_train)

    model = Model(problem.optimizer)
    set_silent(model)

    # 1 Stage
    @variable(model, buy_1[1:problem.n_products] >= 0)

    # 2 Stage
    @variable(model, buy_2[1:problem.n_products, 1:S] >= 0)
    @variable(model, ship[1:problem.n_products, 1:problem.n_clients, 1:S] >= 0)

    for s in 1:S
        d_s = problem.samples_train[s]

        for j in 1:problem.n_clients
            @constraint(model, sum(ship[i,j,s] for i in 1:problem.n_products) >= d_s[j])
        end

        for i in 1:problem.n_products
            @constraint(model, sum(ship[i,j,s] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i,s])
        end
    end

    expr_first = sum(problem.prod_cost_1[i] * buy_1[i] for i in 1:problem.n_products)

    expr_second = 0.0
    for s in 1:S
        p = problem.probs[s]
        expr_second += p * (
            sum(problem.prod_cost_2[i] * buy_2[i,s] for i in 1:problem.n_products) +
            sum(problem.client_cost[i,j] * ship[i,j,s] for i in 1:problem.n_products, j in 1:problem.n_clients)
        )
    end

    @objective(model, Min, expr_first + expr_second)

    optimize!(model)
    first_stage_decision = value.(model[:buy_1])
    obj_value = objective_value(model)

    return SPModel(problem, first_stage_decision, model, obj_value, 0)
end

function sp_eev(
    problem::ShipmentPlanning
)
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, buy_1[1:problem.n_products] .>= 0)
    @variable(model, buy_2[1:problem.n_products] .>= 0)
    @variable(model, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)

    for j in 1:problem.n_clients
        @constraint(model, sum(ship[i, j] for i in 1:problem.n_products) >= Distributions.mean(problem.distribution))
    end
    for i in 1:problem.n_products
        @constraint(model, sum(ship[i, j] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(model, Min,
        sum(problem.prod_cost_1 .* buy_1) +
        sum(problem.prod_cost_2 .* buy_2) +
        sum(sum(problem.client_cost .* ship))
    )

    optimize!(model)
    first_stage_decision = value.(model[:buy_1])
    obj_value = objective_value(model)

    return SPModel(problem, first_stage_decision, model, obj_value, 0)
end

function sp_ws(
    problem::ShipmentPlanning,
    samples_test
)
    # Wait and see
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, buy_1[1:problem.n_products] .>= 0)

    @variable(model, buy_2[1:problem.n_products] .>= 0)

    @variable(model, ship[1:problem.n_products, 1:problem.n_clients] .>= 0)

    for i in 1:problem.n_products
        @constraint(model, sum(ship[i, j] for j in 1:problem.n_clients) <= buy_1[i] + buy_2[i])
    end

    demand_constraints = Vector{ConstraintRef}(undef, problem.n_clients)
    for j in 1:problem.n_clients
        demand_constraints[j] = @constraint(model, sum(ship[i, j] for i in 1:problem.n_products) >= 0.0)
    end

    @objective(model, Min,
                + sum(problem.prod_cost_1 .* buy_1)
                + sum(problem.prod_cost_2 .* buy_2)
                + sum(sum(problem.client_cost .* ship)))
    total = 0
    for sample in samples_test
        for j in 1:problem.n_clients
            set_normalized_rhs(demand_constraints[j], sample[j])
        end
        
        optimize!(model)
        total += objective_value(model)
    end

    return total/length(samples_test)
end