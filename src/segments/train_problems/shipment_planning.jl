function sp_demand_uncertaint_ldr(
    n_products::Int,
    n_clients::Int,
    prod_cost_1::Vector{Float64},
    prod_cost_2::Vector{Float64},
    client_cost::Matrix{Float64},
    distribution,
    optimizer
)

    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    @variable(ldr, buy_1[1:n_products] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, buy_2[1:n_products] .>= 0)
    @variable(ldr, ship[1:n_products, 1:n_clients] .>= 0)

    @variable(ldr, demand[1:n_clients] in LinearDecisionRules.Uncertainty(;
                                    distribution = product_distribution([
                                        distribution for _ in 1:n_clients
                                    ]),
                                    )
                )

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

    return ldr
end

function sp_cost_uncertaint_ldr(
    n_products::Int,
    n_clients::Int,
    prod_cost_1::Vector{Float64},
    demand::Vector{Float64},
    distribution,
    optimizer
)

    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    @variable(ldr, buy_1[1:n_products] .>= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, buy_2[1:n_products] .>= 0)
    @variable(ldr, ship[1:n_products, 1:n_clients] .>= 0)

    @variable(ldr, prod_cost_2[1:n_products] in LinearDecisionRules.Uncertainty(;
                    distribution = product_distribution([distribution for _ in 1:n_products]))
    )

    dist_vec = vec([distribution for i in 1:n_products, j in 1:n_clients])
    @variable(ldr, client_cost[1:(n_products * n_clients)] in 
        LinearDecisionRules.Uncertainty(; distribution = product_distribution(dist_vec))
    )

    for j in 1:n_clients
        @constraint(ldr, sum(ship[i, j] for i in 1:n_products) >= demand[j])
    end
    for i in 1:n_products
        @constraint(ldr, sum(ship[i, j] for j in 1:n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(ldr, Min,
                + sum(prod_cost_1 .* buy_1)
                + sum(prod_cost_2 .* buy_2)
                + sum(client_cost[(i - 1) * n_clients + j] * ship[i, j]
                        for i in 1:n_products, j in 1:n_clients)
                )

    return ldr
end


function sp_demand_uncertaint_ws(
    n_products::Int,
    n_clients::Int,
    prod_cost_1::Vector{Float64},
    prod_cost_2::Vector{Float64},
    client_cost::Matrix{Float64},
    samples_demand_list,
    optimizer
)
    # Wait and see
    model = JuMP.Model(optimizer)
    set_silent(model)

    @variable(model, buy_1[1:n_products] .>= 0)

    @variable(model, buy_2[1:n_products] .>= 0)

    @variable(model, ship[1:n_products, 1:n_clients] .>= 0)

    for i in 1:n_products
        @constraint(model, sum(ship[i, j] for j in 1:n_clients) <= buy_1[i] + buy_2[i])
    end

    demand_constraints = Vector{ConstraintRef}(undef, n_clients)
    for j in 1:n_clients
        demand_constraints[j] = @constraint(model, sum(ship[i, j] for i in 1:n_products) >= 0.0)
    end

    @objective(model, Min,
                + sum(prod_cost_1 .* buy_1)
                + sum(prod_cost_2 .* buy_2)
                + sum(sum(client_cost .* ship)))
    total = 0
    for sample in samples_demand_list
        for j in 1:n_clients
            set_normalized_rhs(demand_constraints[j], sample[j])
        end
        
        optimize!(model)
        total += objective_value(model)
    end

    return total/length(samples_demand_list)
end

function sp_cost_uncertaint_ws(
    n_products::Int,
    n_clients::Int,
    prod_cost_1::Vector{Float64},
    demand::Vector{Float64},
    samples_prod_cost_2,
    samples_client_cost,
    optimizer
)
    # Wait and see
    model = JuMP.Model(optimizer)
    set_silent(model)

    @variable(model, buy_1[1:n_products] .>= 0)

    @variable(model, buy_2[1:n_products] .>= 0)

    @variable(model, ship[1:n_products, 1:n_clients] .>= 0)

    for j in 1:n_clients
         @constraint(model, sum(ship[i, j] for i in 1:n_products) >= demand[j])
    end

    for i in 1:n_products
        @constraint(model, sum(ship[i, j] for j in 1:n_clients) <= buy_1[i] + buy_2[i])
    end

    total = 0
    for (prod_cost_2, client_cost) in zip(samples_prod_cost_2, samples_client_cost)
        @objective(model, Min,
                + sum(prod_cost_1 .* buy_1)
                + sum(prod_cost_2 .* buy_2)
                + sum(sum(client_cost .* ship)))
        
        optimize!(model)
        total += objective_value(model)
    end

    return total/length(samples_prod_cost_2)
end

function combine_cost_sample(
    samples_prod_cost_2,
    samples_client_cost,
    n_samples
)
    prod_cols = collect(samples_prod_cost_2)
    client_cols = collect(samples_client_cost)
    
    combined_samples = [
        vcat(prod_cols[i], vec(client_cols[i]))
        for i in 1:n_samples
    ]

    return combined_samples
end