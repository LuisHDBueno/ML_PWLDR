include("../../pwldr.jl")
include("../train_problems/shipment_planning.jl")
include("../train_problems/shortest_path.jl")

using CSV
using DataFrames

function build_model(ldr_model::LinearDecisionRules.LDRModel,
    n_segments_vec,
    optimizer,
    displace_function,
    dist
    )
    ABC = ldr_model.ext[:_LDR_ABC]
    first_stage_index = ldr_model.ext[:_LDR_first_stage_indices]

    η_min = ABC.lb
    η_max = ABC.ub

    distribution_constructor = (a, b)-> truncated(dist, a, b)
    PWVR_list = displace_function(η_min,
                                    η_max,
                                    ABC,
                                    first_stage_index,
                                    n_segments_vec,
                                    optimizer,
                                    distribution_constructor)

    pwldr = PWLDR(_build_problem(ABC, first_stage_index, PWVR_list, n_segments_vec, optimizer), PWVR_list)
    return pwldr
end

function shipment_planning_test(dist_list, optimizer, n_samples)

    # Fixed because it doesn't affect the uncertainty
    n_products = 10
    n_clients_list = [2, 5, 10]
    n_segments = [2, 5, 10]

    displace_function = [#("black_box", black_box),
                        ("ls_independent", local_search_independent),
                        ("ls_greed", local_search_greed)]

    checkpoint_file = "data/shipment.csv"
    
    if isfile(checkpoint_file)
        results_df = CSV.read(checkpoint_file, DataFrame)
    else
        results_df = DataFrame(
            dist = String[],
            n_clients = Int[],
            n_segments = Int[],
            displace_func = String[],
            PI = Float64[],
            sum_model = Float64[],
            sum_pi = Float64[]
        )
    end

    for (func_name, func) in displace_function
        for (dist_name, dist) in dist_list
            for n_clients in n_clients_list

                #LDR Problem
                ldr, prod_cost_1, prod_cost_2, client_cost = shipment_planning(n_products, n_clients, dist, optimizer)
                optimize!(ldr)

                #Perfect Info sum
                samples_list = eachcol(rand(dist, n_clients, n_samples))
                sum_pi = 0.0
                for sample in samples_list
                    sum_pi += shipment_PI(prod_cost_1, prod_cost_2, client_cost, sample, optimizer)
                end

                for seg in n_segments
                    if any((results_df.dist .== dist_name) .&
                        (results_df.n_clients .== n_clients) .&
                        (results_df.displace_func .== func_name) .&
                        (results_df.n_segments .== seg))
                        println("Already Processed dist=$dist_name, n_clients=$n_clients, func=$func_name, n_segments=$seg")
                        continue
                    end
                    
                    n_segments_vec = fill(seg, n_clients)
                    pwldr = build_model(ldr, n_segments_vec, optimizer, func, dist)
                    optimize!(pwldr)
                    
                    C = value.(pwldr.model.ext[:C])
                    X = value.(pwldr.model[:X])
                    PWVR_list = pwldr.PWVR_list

                    sum_model = 0.0
                    for sample in samples_list
                        sum_model += evaluate_sample(PWVR_list, X, C, sample)
                    end
                    PI = (sum_model - sum_pi) / sum_pi

                    push!(results_df, (
                        dist = dist_name,
                        n_clients = n_clients,
                        n_segments = seg,
                        displace_func = func_name,
                        PI = PI,
                        sum_model = sum_model,
                        sum_pi = sum_pi
                    ))
                    println("Checkpoint: dist=$dist_name, n_clients=$n_clients, func=$func_name, n_segments=$seg")
                    CSV.write(checkpoint_file, results_df)
                end
            end
        end
    end
end

function shortest_path_test(dist_list, optimizer, n_samples)

    n_nodes_list = [5, 10, 25]
    n_edges_list = [15, 30, 75]
    n_segments = [2, 5, 10]
    
    displace_function = [("black_box", black_box),
                        ("local_search_independent", local_search_independent),
                        ("local_search_greed", local_search_greed)]

    checkpoint_file = "data/shortest_path.csv"
    
    if isfile(checkpoint_file)
        results_df = CSV.read(checkpoint_file, DataFrame)
    else
        results_df = DataFrame(
            dist = String[],
            n_nodes = Int[],
            n_edges = Int[],
            n_segments = Int[],
            displace_func = String[],
            PI = Float64[],
            sum_model = Float64[],
            sum_pi = Float64[]
        )
    end

    for (func_name, func) in displace_function
        for (dist_name, dist) in dist_list
            for (n_nodes, n_edges) in zip(n_nodes_list, n_edges_list)

                #LDR Problem
                ldr, A = shortest_path(n_nodes, n_edges, 1, n_nodes, dist, optimizer)
                optimize!(ldr)

                #Perfect Info sum
                samples_list = eachcol(rand(dist, n_edges, n_samples))
                sum_pi = 0.0
                for sample in samples_list
                    sum_pi += shortest_path_PI(A, sample, 1, n_nodes, optimizer)
                end

                for seg in n_segments
                    if any((results_df.dist .== dist_name) .&
                        (results_df.n_nodes .== n_nodes) .&
                        (results_df.n_edges .== n_edges) .&
                        (results_df.displace_func .== func_name) .&
                        (results_df.n_segments .== seg))
                        println("Already Processed dist=$dist_name, n_nodes=$n_nodes, func=$func_name, n_segments=$seg")
                        continue
                    end
                    
                    n_segments_vec = fill(seg, n_edges)
                    pwldr = build_model(ldr, n_segments_vec, optimizer, func, dist)
                    optimize!(pwldr)
                    
                    C = value.(pwldr.model.ext[:C])
                    X = value.(pwldr.model[:X])
                    PWVR_list = pwldr.PWVR_list

                    sum_model = 0.0
                    for sample in samples_list
                        sum_model += evaluate_sample(PWVR_list, X, C, sample)
                    end
                    PI = (sum_model - sum_pi) / sum_pi

                    push!(results_df, (
                        dist = dist_name,
                        n_nodes = n_nodes,
                        n_edges = n_edges,
                        n_segments = seg,
                        displace_func = func_name,
                        PI = PI,
                        sum_model = sum_model,
                        sum_pi = sum_pi
                    ))
                    println("Checkpoint: dist=$dist_name, n_clients=$n_clients, func=$func_name, n_segments=$seg")
                    CSV.write(checkpoint_file, results_df)
                end
            end
        end
    end
end

Random.seed!(1234)

dist_list = [("Unif 10,90", Uniform(10, 90)),
            ("Unif 35,65", Uniform(35, 65)),
            ("Normal 50,5 - 10,90", truncated(Normal(50, 5), 10, 90)),
            ("Normal 50,5 - 35,65", truncated(Normal(50, 5), 35, 65)),
            #("Normal 50,20 - 10,90", truncated(Normal(50, 20), 10, 90)),
            #("Normal 50,20 - 35,65", truncated(Normal(50, 20), 35, 65))
            ]

using HiGHS

shipment_planning_test(dist_list, HiGHS.Optimizer, 200)
#shortest_path_test(dist_list, HiGHS.Optimizer, 200)