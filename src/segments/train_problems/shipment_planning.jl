function shipment_planning_ldr(n_products::Int, n_clients::Int, distribution, optimizer)
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    prod_cost_1 = rand(Uniform(25, 75), n_products)
    @variable(ldr, buy_1[1:n_products] .>= 0, LinearDecisionRules.FirstStage)

    prod_cost_2 = prod_cost_1 + rand(Uniform(10, 25), n_products)
    @variable(ldr, buy_2[1:n_products] .>= 0)

    client_cost = rand(Uniform(25, 100), n_products, n_clients)
    @variable(ldr, sell[1:n_products, 1:n_clients] .>= 0)

    @variable(ldr, demand[1:n_clients] in LinearDecisionRules.Uncertainty(;
                                    distribution = product_distribution([
                                        distribution for _ in 1:n_clients
                                    ]),
                                    )
                )

    # restrição: atender a demanda de cada cliente
    for j in 1:n_clients
        @constraint(ldr, sum(sell[i, j] for i in 1:n_products) >= demand[j])
    end
    for i in 1:n_products
        @constraint(ldr, sum(sell[i, j] for j in 1:n_clients) <= buy_1[i] + buy_2[i])
    end

    @objective(ldr, Min,
                + sum(prod_cost_1 .* buy_1)
                + sum(prod_cost_2 .* buy_2)
                + sum(sum(client_cost .* sell)))

    return ldr, prod_cost_1, prod_cost_2, client_cost
end

function shipment_planning_ws(n_products, n_clients, prod_cost_1, prod_cost_2, client_cost, samples_list, optimizer)
    # Wait and see
    model = JuMP.Model(optimizer)
    set_silent(model)

    @variable(model, buy_1[1:n_products] .>= 0)

    @variable(model, buy_2[1:n_products] .>= 0)

    @variable(model, sell[1:n_products, 1:n_clients] .>= 0)

    for i in 1:n_products
        @constraint(model, sum(sell[i, j] for j in 1:n_clients) <= buy_1[i] + buy_2[i])
    end

    demand_constraints = Vector{ConstraintRef}(undef, n_clients)
    for j in 1:n_clients
        demand_constraints[j] = @constraint(model, sum(sell[i, j] for i in 1:n_products) >= 0.0)
    end

    @objective(model, Min,
                + sum(prod_cost_1 .* buy_1)
                + sum(prod_cost_2 .* buy_2)
                + sum(sum(client_cost .* sell)))
    total = 0
    for sample in samples_list
        for j in 1:n_clients
            set_normalized_rhs(demand_constraints[j], sample[j])
        end
        
        optimize!(model)
        total += objective_value(model)
    end

    return total/length(samples_list)
end