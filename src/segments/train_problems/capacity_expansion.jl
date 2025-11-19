struct CapacityExpansionMetadata
    n_plants::Int
    periods::Int
    invest_cost::Vector{Float64}
    penality_cost::Vector{Float64}
    demand_dist::Vector{Distribution{Univariate, Continuous}}
    oper_cost::Vector{Float64}
    min_gen_fraction::Vector{Float64}
    max_budget::Float64
    samples_train
    samples_test
    optimizer
end

function ce_second_stage(
    problem_inst::ProblemInstance,
    samples_test
)
    problem = problem_inst.metadata
    # Solve with fixed first stage
    fixed_cost = sum(problem.invest_cost .* problem_inst.first_stage_decision)

    recourse_model = JuMP.Model(problem.optimizer)
    set_silent(recourse_model)

    @variable(recourse_model, gen[1:problem.n_plants, 1:problem.periods] .>= 0)
    @variable(recourse_model, shortfall[1:problem.periods] .>= 0)

    demand_constraint = Vector{ConstraintRef}(undef, problem.periods)
    for j in 1:problem.periods
        demand_constraint[j] = @constraint(recourse_model,
            sum(gen[1:end, j]) + shortfall[j] >= 0.0)

        for i in 1:problem.n_plants
            @constraint(recourse_model, gen[i,j] <= problem_inst.first_stage_decision[i])
            @constraint(recourse_model, gen[i,j] >= problem.min_gen_fraction[i] * problem_inst.first_stage_decision[i])
        end
    end

    @objective(recourse_model, Min,
                + sum(problem.penality_cost .* shortfall)
                + sum(sum(gen[i, 1:end] .* problem.oper_cost[i]) for i in 1:problem.n_plants)
                )

    total_cost = 0.0
    for sample in problem.samples_test
        for j in 1:problem.periods
            set_normalized_rhs(demand_constraint[j], sample[j])
        end
        optimize!(recourse_model)
        scenario_cost = fixed_cost + objective_value(recourse_model)
        total_cost += scenario_cost
    end

    return total_cost / length(samples_test)
end

function ce_ldr(
    problem::CapacityExpansionMetadata
)
    ldr = LinearDecisionRules.LDRModel(problem.optimizer)
    set_silent(ldr)

    @variable(ldr, build[1:problem.n_plants] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, gen[1:problem.n_plants, 1:problem.periods] .>= 0)
    @variable(ldr, shortfall[1:problem.periods] .>= 0)

    @variable(ldr, demand[i = 1:problem.n_plants] 
                in LinearDecisionRules.ScalarUncertainty(problem.demand_dist[i]))

    @constraint(ldr, sum(problem.invest_cost .* build) <= problem.max_budget)
    for j in 1:problem.periods

        @constraint(ldr, sum(gen[1:end, j]) + shortfall[j] >= demand[j])

        for i in 1:problem.n_plants
            @constraint(ldr, gen[i,j] <= build[i])
            @constraint(ldr, gen[i,j] >= problem.min_gen_fraction[i] * build[i])
        end
    end

    @objective(ldr, Min,
                + sum(problem.invest_cost .* build)
                + sum(problem.penality_cost .* shortfall)
                + sum(sum(gen[i, 1:end] .* problem.oper_cost[i]) for i in 1:problem.n_plants)
                )
    optimize!(ldr)

    first_stage_decision = [LinearDecisionRules.get_decision(ldr, build[i]) for i in 1:problem.n_plants]
    obj_value = objective_value(ldr)

    return ProblemInstance(problem, first_stage_decision, ldr, obj_value, 0)
end

function ce_standard_form(
    problem::CapacityExpansionMetadata
)
    S = length(problem.samples_train)

    model = Model(problem.optimizer)
    set_silent(model)

    # 1 Stage
    @variable(model, build[1:problem.n_plants] .>= 0)

    # 2 Stage
    @variable(model, gen[1:problem.n_plants, 1:problem.periods, 1:S] .>= 0)
    @variable(model, shortfall[1:problem.periods, 1:S] .>= 0)

    @constraint(model, sum(problem.invest_cost .* build) <= problem.max_budget)
    for (s, sample) in enumerate(problem.samples_train)
        for j in 1:problem.periods
            @constraint(model, sum(gen[1:end, j, s]) + shortfall[j, s] >= sample[j])
            for i in 1:problem.n_plants
                @constraint(model, gen[i,j,s] <= build[i])
                @constraint(model, gen[i,j,s] >= problem.min_gen_fraction[i] * build[i])
            end
        end
    end

    expr_first = sum(problem.invest_cost .* build)

    expr_second = 0.0
    for s in 1:S
        p = 1.0/S
        expr_second += p * (
            sum(problem.penality_cost .* shortfall[1:end, s])
            + sum(sum(gen[i, 1:end, s] .* problem.oper_cost[i]) for i in 1:problem.n_plants)
        )
    end

    @objective(model, Min, expr_first + expr_second)

    optimize!(model)
    first_stage_decision = value.(model[:build])
    obj_value = objective_value(model)

    return ProblemInstance(problem, first_stage_decision, model, obj_value, 0)
end

function ce_deterministic(
    problem::CapacityExpansionMetadata
)
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, build[1:problem.n_plants] .>= 0)
    @variable(model, gen[1:problem.n_plants, 1:problem.periods] .>= 0)
    @variable(model, shortfall[1:problem.periods] .>= 0)

    @constraint(model, sum(problem.invest_cost .* build) <= problem.max_budget)
    for j in 1:problem.periods
        demand_j = Distributions.mean(problem.demand_dist[j])
        @constraint(model,
            sum(gen[1:end, j]) + shortfall[j] >= demand_j)
        for i in 1:problem.n_plants
            @constraint(model, gen[i,j] <= build[i])
            @constraint(model, gen[i,j] >= problem.min_gen_fraction[i] * build[i])
        end
    end

    @objective(model, Min,
                + sum(problem.invest_cost .* build)
                + sum(problem.penality_cost .* shortfall)
                + sum(sum(gen[i, 1:end] .* problem.oper_cost[i]) for i in 1:problem.n_plants)
                )
    optimize!(model)

    first_stage_decision = value.(model[:build])
    obj_value = objective_value(model)

    return ProblemInstance(problem, first_stage_decision, model, obj_value, 0)

end

function ce_ws(
    problem::CapacityExpansionMetadata
)
    samples_test = problem.samples_test
    # Wait and see
    model = JuMP.Model(problem.optimizer)
    set_silent(model)

    @variable(model, build[1:problem.n_plants] .>= 0)
    @variable(model, gen[1:problem.n_plants, 1:problem.periods] .>= 0)
    @variable(model, shortfall[1:problem.periods] .>= 0)

    @constraint(model, sum(problem.invest_cost .* build) <= problem.max_budget)
    demand_constraint = Vector{ConstraintRef}(undef, problem.periods)
    for j in 1:problem.periods
        demand_constraint[j] = @constraint(model, sum(gen[1:end, j]) + shortfall[j]>= 0.0)
        for i in 1:problem.n_plants
            @constraint(model, gen[i,j] <= build[i])
            @constraint(model, gen[i,j] >= problem.min_gen_fraction[i] * build[i])
        end
    end

    @objective(model, Min, 0.0)
    expr_first = sum(problem.invest_cost .* build)
    total_cost = 0.0
    for sample in problem.samples_test
        for j in 1:problem.periods
            set_normalized_rhs(demand_constraint[j], sample[j])
        end
    
        expr_second = sum(problem.penality_cost .* shortfall)
                    + sum(sum(gen[i, 1:end] .* problem.oper_cost[i]) for i in 1:problem.n_plants)

        set_objective_function(model, expr_first + expr_second)

        optimize!(model)
        total_cost += objective_value(model)
    end

    return total_cost / length(samples_test)
end

function ce_gen_metadata(
    dist_list::Vector{Distribution{Univariate, Continuous}},
    n_samples_train::Int,
    n_samples_test::Int,
    optimizer
)
    periods = size(dist_list, 1)
    n_plants = size(dist_list, 1)

    invest_cost = rand(Uniform(500, 1000), n_plants)
    penality_cost = fill(1500, periods)
    oper_cost = rand(Uniform(50, 100), n_plants)

    min_gen_fraction = rand(Uniform(0.2, 0.5), n_plants)

    avg_demand_per_period = sum(mean.(dist_list))

    avg_cost_build_all = sum(invest_cost) * (avg_demand_per_period / n_plants)
    max_budget = avg_cost_build_all * 1.25

    demand_dist = shuffle(dist_list)

    samples_train = [rand.(demand_dist) for _ in 1:n_samples_train]
    samples_test = [rand.(demand_dist) for _ in 1:n_samples_test]

    return CapacityExpansionMetadata(
        n_plants, periods, invest_cost, penality_cost, demand_dist, oper_cost,
        min_gen_fraction, max_budget, samples_train, samples_test, optimizer
    )
end

CapacityExpansionSetup = ProblemSetup("capacity_expansion",
                            ce_gen_metadata, ce_second_stage, ce_ldr, ce_ws,
                            ce_standard_form, ce_deterministic
                            )