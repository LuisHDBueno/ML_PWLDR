include("../../pwldr.jl")
include("../train_problems/shipment_planning.jl")

#Models
include("black_box.jl")
include("local_search.jl")

using CSV
using DataFrames
using HiGHS

function evaluate_ldr(ldr, sample)
    X = value.(ldr.primal_model[:X])
    C = ldr.ext[:_LDR_ABC].C
    ξ = [1.0]
    append!(ξ, sample)
    return (C * ξ)' * (X * ξ)
end

function eev_ldr_calc(ldr, sample_list)
    X = value.(ldr.primal_model[:X])
    C = ldr.ext[:_LDR_ABC].C
    sum = 0
    for sample in sample_list
        ξ = [1.0]
        append!(ξ, sample)
        sum += (C * ξ)' * (X * ξ)
    end
    return sum/length(sample_list)
end

function eev_pwldr_calc(pwldr, sample_list)
    sum = 0
    X = value.(pwldr.model[:X])
    C = pwldr.model.ext[:C]
    for sample in sample_list
        sum += evaluate_sample(pwldr.PWVR_list, X, C, sample)
    end
    return sum/length(sample_list)
end

function shipment_planning_test(dist_list, optimizer, n_samples)

    n_products_list = [5]
    n_clients_list = [2, 3, 4, 5]
    n_segments = [2, 3, 4, 5]

    displace_function = [("black_box", black_box!),
                        ("ls_independent", local_search_independent!),
                        ("ls_dependent", local_search!)]

    checkpoint_file = "data/shipment_planning2.csv"
    
    if isfile(checkpoint_file)
        results_df = CSV.read(checkpoint_file, DataFrame)
    else
        results_df = DataFrame(
            dist = String[],
            n_clients = Int[],
            n_products = Int[],
            n_segments = Int[],
            displace_func = String[],
            uncertaint_type = String[],
            rp_ldr = Float64[],
            rp_pwldr_uniform = Float64[],
            rp_pwldr = Float64[],
            eev_ldr = Float64[],
            eev_pwldr_uniform = Float64[],
            eev_pwldr = Float64[],
            ws = Float64[],
            opt_time = Float64[]
        )
    end

    println("Init shipment planning")
    for (dist_name, dist) in dist_list
        for n_products in n_products_list
            for n_clients in n_clients_list
                prod_cost_1 = rand(Uniform(5, 10), n_products)
                prod_cost_2 = rand(dist, n_products)
                client_cost = rand(dist, n_products, n_clients)
                demand = rand(dist, n_clients)

                samples_demand = eachcol(rand(dist, n_clients, n_samples))
                samples_prod_cost_2 = eachcol(rand(dist, n_products, n_samples))
                samples_client_cost = eachslice(rand(dist, n_products, n_clients, n_samples), dims=3)
                cost_samples = combine_cost_sample(samples_prod_cost_2, samples_client_cost, n_samples)

                # LDR Problem
                ## Demand Uncertainty
                distribution_constructor = (a, b)-> truncated(dist, a, b)
                ldr_demand = sp_demand_uncertaint_ldr(n_products, n_clients, prod_cost_1,
                                                prod_cost_2, client_cost, dist, optimizer)
                optimize!(ldr_demand)

                rp_ldr_demand = objective_value(ldr_demand)
                eev_ldr_demand = eev_ldr_calc(ldr_demand, samples_demand)
                ws_demand = sp_demand_uncertaint_ws(n_products, n_clients, prod_cost_1,
                                                prod_cost_2, client_cost, samples_demand, optimizer)
                
                ## Cost Uncertainty
                """
                ldr_cost = sp_cost_uncertaint_ldr(n_products, n_clients, prod_cost_1,
                                demand, dist, optimizer)
                optimize!(ldr_cost)

                rp_ldr_cost = objective_value(ldr_cost)
                eev_ldr_cost = eev_ldr_calc(ldr_cost, cost_samples)
                ws_cost = sp_cost_uncertaint_ws(n_products, n_clients, prod_cost_1,
                            demand, samples_prod_cost_2, samples_client_cost, optimizer)
                """
                
                #Piecewise PDR Problem
                for seg in n_segments

                    # Uniform segments
                    ## Demand Uncertainty
                    n_segments_vec_demand = _segments_number(ldr_demand; fix_n = seg)
                    pwldr_model_demand = PWLDR(ldr_demand, optimizer, distribution_constructor,
                                                n_segments_vec_demand)
                    optimize!(pwldr_model_demand)

                    rp_pwldr_uniform_demand = objective_value(pwldr_model_demand)
                    eev_pwldr_uniform_demand = eev_pwldr_calc(pwldr_model_demand, samples_demand)

                    ## Cost Uncertainty
                    """
                    n_segments_vec_cost = _segments_number(ldr_cost; fix_n = seg)
                    pwldr_model_cost = PWLDR(ldr_cost, optimizer, distribution_constructor,
                                                n_segments_vec_cost)
                    optimize!(pwldr_model_cost)

                    rp_pwldr_uniform_cost = objective_value(pwldr_model_cost)
                    eev_pwldr_uniform_cost = eev_pwldr_calc(pwldr_model_cost, cost_samples)
                    """

                    # Displace functions
                    for (func_name, func) in displace_function
                        init_time_demand = time()
                        func(pwldr_model_demand)
                        end_time_demand = time()
                        
                        optimize!(pwldr_model_demand)
                        rp_pwldr_demand = objective_value(pwldr_model_demand)
                        eev_pwldr_demand = eev_pwldr_calc(pwldr_model_demand, samples_demand)

                        push!(results_df, (
                                dist = dist_name,
                                n_clients = n_clients,
                                n_products = n_products,
                                n_segments = seg,
                                displace_func = func_name,
                                uncertaint_type = "demand",
                                rp_ldr = rp_ldr_demand,
                                rp_pwldr_uniform = rp_pwldr_uniform_demand,
                                rp_pwldr = rp_pwldr_demand,
                                eev_ldr = eev_ldr_demand,
                                eev_pwldr_uniform = eev_pwldr_uniform_demand,
                                eev_pwldr = eev_pwldr_demand,
                                ws = ws_demand,
                                opt_time = end_time_demand - init_time_demand
                            ), promote = true)
                        """
                        init_time_cost = time()
                        func(pwldr_model_cost)
                        end_time_cost = time()
                        
                        optimize!(pwldr_model_cost)
                        rp_pwldr_cost = objective_value(pwldr_model_cost)
                        eev_pwldr_cost = eev_pwldr_calc(pwldr_model_cost, cost_samples)

                        push!(results_df, (
                                dist = dist_name,
                                n_clients = n_clients,
                                n_products = n_products,
                                n_segments = seg,
                                displace_func = func_name,
                                uncertaint_type = "cost",
                                rp_ldr = rp_ldr_cost,
                                rp_pwldr_uniform = rp_pwldr_uniform_cost,
                                rp_pwldr = rp_pwldr_cost,
                                eev_ldr = eev_ldr_cost,
                                eev_pwldr_uniform = eev_pwldr_uniform_cost,
                                eev_pwldr = eev_pwldr_cost,
                                ws = ws_cost,
                                opt_time = end_time_cost - init_time_cost
                            ), promote = true)
                        """
                        
                        println("Checkpoint: dist=$dist_name, n_clients=$n_clients, func=$func_name, n_segments=$seg")
                        CSV.write(checkpoint_file, results_df)
                    end
                end
            end
        end
    end
end

Random.seed!(1234)

dist_list = [("Unif 10,90", Uniform(10, 90)),
            ("Normal 50,5 - 10,90", truncated(Normal(50, 5), 10, 90)),
            ("Normal 50,5 - 35,65", truncated(Normal(50, 5), 35, 65)),
            ]

shipment_planning_test(dist_list, HiGHS.Optimizer, 1000)