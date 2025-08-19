include("../../pwldr.jl")
include("../train_problems/shipment_planning.jl")
include("../train_problems/shortest_path.jl")

#Models
include("black_box.jl")
include("local_search.jl")

using CSV
using DataFrames

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

    # Fixed because it doesn't affect the uncertainty
    n_products = 10
    n_clients_list = [2, 5]
    n_segments = [2, 3, 4, 5]

    displace_function = [("black_box", black_box!),
                        ("ls_independent", local_search_independent!)]

    checkpoint_file = "data/shipment.csv"
    
    if isfile(checkpoint_file)
        results_df = CSV.read(checkpoint_file, DataFrame)
    else
        results_df = DataFrame(
            dist = String[],
            n_clients = Int[],
            n_segments = Int[],
            displace_func = String[],
            rp_ldr = Float64[],
            rp_pwldr = Float64[],
            eev_ldr = Float64[],
            eev_pwldr = Float64[],
            ws = Float64[],
            opt_time = Float64[]
        )
    end

    println("Init shipment planning")
    for (func_name, func) in displace_function
        for (dist_name, dist) in dist_list
            for n_clients in n_clients_list
                # Evaluate metrics at orginial model
                samples_list = eachcol(rand(dist, n_clients, n_samples))
            
                #LDR Problem
                distribution_constructor = (a, b)-> truncated(dist, a, b)
                ldr, prod_cost_1, prod_cost_2, client_cost = shipment_planning_ldr(n_products, n_clients, dist, optimizer)
                optimize!(ldr)

                rp_ldr = objective_value(ldr)
                eev_ldr = eev_ldr_calc(ldr, samples_list)
                ws = shipment_planning_ws(n_products, n_clients, prod_cost_1, prod_cost_2, client_cost, samples_list, optimizer)
                
                #Piecewise PDR Problem
                for seg in n_segments
                    if any((results_df.dist .== dist_name) .&
                        (results_df.n_clients .== n_clients) .&
                        (results_df.displace_func .== func_name) .&
                        (results_df.n_segments .== seg))
                        println("Already Processed dist=$dist_name, n_clients=$n_clients, func=$func_name, n_segments=$seg")
                        continue
                    end  
                    n_segments_vec = _segments_number(ldr; fix_n = seg)
                    pwldr_model = PWLDR(ldr, optimizer, distribution_constructor, n_segments_vec)
                    init_time = time()
                    func(pwldr_model)
                    end_time = time()

                    optimize!(pwldr_model)
                    rp_pwldr = objective_value(pwldr_model)

                    eev_pwldr = eev_pwldr_calc(pwldr_model, samples_list)

                    push!(results_df, (
                        dist = dist_name,
                        n_clients = n_clients,
                        n_segments = seg,
                        displace_func = func_name,
                        rp_ldr = rp_ldr,
                        rp_pwldr = rp_pwldr,
                        eev_ldr = eev_ldr,
                        eev_pwldr = eev_pwldr,
                        ws = ws,
                        opt_time = end_time - init_time
                    ), promote = true)
                    println("Checkpoint: dist=$dist_name, n_clients=$n_clients, func=$func_name, n_segments=$seg")
                    CSV.write(checkpoint_file, results_df)
                end
            end
        end
    end
end

function shortest_path_test(dist_list, optimizer, n_samples)

    n_nodes_list = [5, 10]
    n_edges_list = [15, 30]
    n_segments = [2, 3, 4, 5]
    
    displace_function = [("black_box", black_box!),
                        ("local_search_independent", local_search_independent!)]

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
            rp_ldr = Float64[],
            rp_pwldr = Float64[],
            eev_ldr = Float64[],
            eev_pwldr = Float64[],
            ws = Float64[],
            opt_time = Float64[]
        )
    end

    println("Init shortest path")
    for (func_name, func) in displace_function
        for (dist_name, dist) in dist_list
            for (n_nodes, n_edges) in zip(n_nodes_list, n_edges_list)
                # Evaluate metrics at orginial model
                samples_list = eachcol(rand(dist, n_edges, n_samples))

                #LDR Problem
                distribution_constructor = (a, b)-> truncated(dist, a, b)
                ldr, A = shortest_path_ldr(n_nodes, n_edges, 1, n_nodes, dist, optimizer)
                optimize!(ldr)

                rp_ldr = objective_value(ldr)
                eev_ldr = eev_ldr_calc(ldr, samples_list)
                ws = shortest_path_ws(A, n_edges, n_nodes, samples_list, 1, n_nodes, optimizer)

                #Piecewise PDR Problem
                for seg in n_segments
                    if any((results_df.dist .== dist_name) .&
                        (results_df.n_nodes .== n_nodes) .&
                        (results_df.n_edges .== n_edges) .&
                        (results_df.displace_func .== func_name) .&
                        (results_df.n_segments .== seg))
                        println("Already Processed dist=$dist_name, n_nodes=$n_nodes, func=$func_name, n_segments=$seg")
                        continue
                    end
                    
                    n_segments_vec = _segments_number(ldr; fix_n = seg)
                    pwldr_model = PWLDR(ldr, optimizer, distribution_constructor, n_segments_vec)
                    init_time = time()
                    func(pwldr_model)
                    end_time = time()
                    optimize!(pwldr_model)
                    rp_pwldr = objective_value(pwldr_model)

                    eev_pwldr = eev_pwldr_calc(pwldr_model, samples_list)

                    push!(results_df, (
                        dist = dist_name,
                        n_nodes = n_nodes,
                        n_edges = n_edges,
                        n_segments = seg,
                        displace_func = func_name,
                        rp_ldr = rp_ldr,
                        rp_pwldr = rp_pwldr,
                        eev_ldr = eev_ldr,
                        eev_pwldr = eev_pwldr,
                        ws = ws,
                        opt_time = end_time - init_time
                    ), promote = true)
                    println("Checkpoint: dist=$dist_name, n_edges=$n_edges, n_nodes=$n_nodes, func=$func_name, n_segments=$seg")
                    CSV.write(checkpoint_file, results_df)
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

using HiGHS

shipment_planning_test(dist_list, HiGHS.Optimizer, 1000)
shortest_path_test(dist_list, HiGHS.Optimizer, 1000)