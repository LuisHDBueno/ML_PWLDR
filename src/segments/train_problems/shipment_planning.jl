function shipment_planning(n_products::Int, n_clients::Int, distribution, optimizer)
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    prod_cost_1 = rand(Uniform(25, 75), n_products)
    @variable(ldr, buy_1[1:n_products] .>= 0, LinearDecisionRules.FirstStage)

    prod_cost_2 = prod_cost_1 + rand(Uniform(10, 25), n_products)
    @variable(ldr, buy_2[1:n_products] .>= 0)

    client_cost = rand(Uniform(25, 100), n_products, n_clients)
    @variable(ldr, sell[1:n_products, 1:n_clients] .>= 0)

    dists = [distribution for _ in 1:n_clients]
    joint_dist = Product(dists)
    @variable(ldr, demand[1:n_clients] in LinearDecisionRules.Uncertainty(
        distribution = joint_dist
    ))

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

function shipment_PI(prod_cost_1, prod_cost_2, client_cost, demand, optimizer)
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    n_products = length(prod_cost_1)
    n_clients = size(client_cost, 2)

    @variable(ldr, buy_1[1:n_products] .>= 0, LinearDecisionRules.FirstStage)

    @variable(ldr, buy_2[1:n_products] .>= 0, LinearDecisionRules.FirstStage)

    @variable(ldr, sell[1:n_products, 1:n_clients] .>= 0, LinearDecisionRules.FirstStage)

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

    optimize!(ldr)

    return objective_value(ldr)
end