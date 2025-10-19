include("../../pwldr.jl")
include("../train_problems/shipment_planning.jl")
include("../train_problems/network_flow_planning.jl")

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

function dr_ldr_calc(ldr, sample_list)
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

function dr_pwldr_calc(pwldr, sample_list)
    sum = 0
    X = value.(pwldr.model[:X])
    C = pwldr.model.ext[:C]
    for sample in sample_list
        sum += evaluate_sample(pwldr.PWVR_list, X, C, sample)
    end
    return sum/length(sample_list)
end

function shipment_planning_test(
    dist_list,
    optimizer,
    n_samples_train,
    n_samples_test,
    n_problems)

    n_products = 5
    n_clients = 5
    n_segments = [2, 3, 4, 5, 6]#, 7, 8, 9, 10]

    displace_function = [("black_box", black_box!),
                        ("ls_independent", local_search_independent!),
                        ("ls_dependent", local_search!)]

    checkpoint_file = "data/shipment_planning2.csv"
    
    if isfile(checkpoint_file)
        results_df = CSV.read(checkpoint_file, DataFrame)
    else
        results_df = DataFrame(
            dist = String[],
            idx_problem = Int[],
            n_segments = Int[],
            displace_func = String[],
            reoptm_standard_form = Float64[],
            reoptm_ldr = Float64[],
            reoptm_pwldr_uniform = Float64[],
            reoptm_pwldr_optimize = Float64[],
            reoptm_eev = Float64[],
            dr_ldr = Float64[],
            dr_pwldr_uniform = Float64[],
            dr_pwldr_optimize = Float64[],
            obj_value_ldr = Float64[],
            obj_value_pwldr_uniform = Float64[],
            obj_value_pwldr_optimize = Float64[],
            obj_value_eev = Float64[],
            obj_value_standard_form = Float64[],
            ws = Float64[],
            opt_time = Float64[]
        )
    end

    println("Init shipment planning")
    for idx in 1:n_problems
        prod_cost_1 = rand(Uniform(50, 100), n_products)
        prod_cost_2 = prod_cost_1 .+ rand(Uniform(99, 100), n_products)
        client_cost = rand(Uniform(25, 50), n_products, n_clients)
        probs = fill(1.0/n_samples_train, n_samples_train)
        
        for (dist_name, dist) in dist_list

            samples_demand_train = eachcol(rand(dist, n_clients, n_samples_train))
            samples_demand_test = eachcol(rand(dist, n_clients, n_samples_test))
            @show size(samples_demand_test)

            problem = ShipmentPlanning(
                n_products,
                n_clients,
                prod_cost_1,
                prod_cost_2,
                client_cost,
                samples_demand_train,
                probs,
                dist,
                optimizer
            )

            standard_form_model = sp_standard_form(problem)
            eev_model = sp_eev(problem)

            obj_value_standard_form = standard_form_model.objective_value
            reoptm_standard_form = sp_second_stage(standard_form_model, samples_demand_test)

            obj_value_eev = eev_model.objective_value
            reoptm_eev = sp_second_stage(eev_model, samples_demand_test)

            ws = sp_ws(problem, samples_demand_test)

            # LDR Problem
            ## Demand Uncertainty
            distribution_constructor = (a, b)-> truncated(dist, a, b)
            ldr_model = sp_ldr(problem, dist)

            reoptm_ldr = sp_second_stage(ldr_model, samples_demand_test)
            dr_ldr = dr_ldr_calc(ldr_model.model, samples_demand_test)
            obj_value_ldr = ldr_model.objective_value

            ldr = ldr_model.model
            
            #Piecewise PDR Problem
            for seg in n_segments

                # Uniform segments
                ## Demand Uncertainty
                n_segments_vec = _segments_number(ldr; fix_n = seg)
                pwldr = PWLDR(ldr, optimizer,
                                distribution_constructor, n_segments_vec)

                pwldr_model = SPModelPWLDR(problem, pwldr)
                reoptm_pwldr_uniform = sp_second_stage(pwldr_model, samples_demand_test)
                dr_pwldr_uniform = dr_pwldr_calc(pwldr_model.model, samples_demand_test)
                obj_value_pwldr_uniform = pwldr_model.objective_value

                # Displace functions
                for (func_name, func) in displace_function
                    init_time = time()
                    func(pwldr)
                    end_time = time()
                    
                    pwldr_model = SPModelPWLDR(problem, pwldr)

                    reoptm_pwldr_optimize = sp_second_stage(pwldr_model, samples_demand_test)
                    dr_pwldr_optimize = dr_pwldr_calc(pwldr_model.model, samples_demand_test)
                    obj_value_pwldr_optimize = pwldr_model.objective_value

                    push!(results_df, (
                            dist = dist_name,
                            idx_problem = idx,
                            n_segments = seg,
                            displace_func = func_name,
                            reoptm_standard_form = reoptm_standard_form,
                            reoptm_ldr = reoptm_ldr,
                            reoptm_pwldr_uniform = reoptm_pwldr_uniform,
                            reoptm_pwldr_optimize = reoptm_pwldr_optimize,
                            reoptm_eev = reoptm_eev,
                            dr_ldr = dr_ldr,
                            dr_pwldr_uniform = dr_pwldr_uniform,
                            dr_pwldr_optimize = dr_pwldr_optimize,
                            obj_value_ldr = obj_value_ldr,
                            obj_value_pwldr_uniform = obj_value_pwldr_uniform,
                            obj_value_pwldr_optimize = obj_value_pwldr_optimize,
                            obj_value_eev = obj_value_eev,
                            obj_value_standard_form = obj_value_standard_form,
                            ws = ws,
                            opt_time = end_time - init_time
                        ), promote = true)
                    
                    println("Checkpoint: dist=$dist_name, func=$func_name, n_segments=$seg")
                    CSV.write(checkpoint_file, results_df)
                end
            end
        end
    end
end

function network_flow_planning_test(
    dist_list,
    optimizer,
    n_samples_train,
    n_samples_test,
    n_problems)

    n_nodes = 5
    n_edges = 7
    n_commodities = 3
    n_segments = [2, 3, 4, 5, 6]#, 7, 8, 9, 10]

    displace_function = [("black_box", black_box!),
                        ("ls_independent", local_search_independent!),
                        ("ls_dependent", local_search!)]

    checkpoint_file = "data/network_flow_planning.csv"

    if isfile(checkpoint_file)
        results_df = CSV.read(checkpoint_file, DataFrame)
    else
        results_df = DataFrame(
            dist = String[],
            idx_problem = Int[],
            n_segments = Int[],
            displace_func = String[],
            reoptm_standard_form = Float64[],
            reoptm_ldr = Float64[],
            reoptm_pwldr_uniform = Float64[],
            reoptm_pwldr_optimize = Float64[],
            reoptm_eev = Float64[],
            dr_ldr = Float64[],
            dr_pwldr_uniform = Float64[],
            dr_pwldr_optimize = Float64[],
            obj_value_ldr = Float64[],
            obj_value_pwldr_uniform = Float64[],
            obj_value_pwldr_optimize = Float64[],
            obj_value_eev = Float64[],
            obj_value_standard_form = Float64[],
            ws = Float64[],
            opt_time = Float64[]
        )
    end

    println("Init network flow planning")
    for idx in 1:n_problems
        for (dist_name, dist) in dist_list
            problem, samples_demand_test, samples_cost_test = generate_networkflow_problem(
                                                                            n_nodes, n_edges, n_commodities,
                                                                            n_samples_train, n_samples_test,
                                                                            dist, optimizer)
            samples_test = collect(eachcol(vcat(samples_demand_test, samples_cost_test)))

            standard_form_model = nfp_standard_form(problem)
            eev_model = nfp_eev(problem)

            obj_value_standard_form = standard_form_model.objective_value
            reoptm_standard_form = nfp_second_stage(standard_form_model, samples_demand_test, samples_cost_test)

            obj_value_eev = eev_model.objective_value
            reoptm_eev = nfp_second_stage(eev_model, samples_demand_test, samples_cost_test)

            ws = nfp_ws(problem, samples_demand_test, samples_cost_test)

            # LDR Problem
            distribution_constructor = (a, b)-> truncated(dist, a, b)
            ldr_model = nfp_ldr(problem, dist)

            reoptm_ldr = nfp_second_stage(ldr_model, samples_demand_test, samples_cost_test)
            dr_ldr = dr_ldr_calc(ldr_model.model, samples_test)
            obj_value_ldr = ldr_model.objective_value

            ldr = ldr_model.model
            
            #Piecewise PDR Problem
            for seg in n_segments

                # Uniform segments
                n_segments_vec = _segments_number(ldr; fix_n = seg)
                pwldr = PWLDR(ldr, optimizer,
                                distribution_constructor, n_segments_vec)

                pwldr_model = NFPModelPWLDR(problem, pwldr)
                reoptm_pwldr_uniform = nfp_second_stage(pwldr_model, samples_demand_test, samples_cost_test)
                dr_pwldr_uniform = dr_pwldr_calc(pwldr_model.model, samples_test)
                obj_value_pwldr_uniform = pwldr_model.objective_value

                # Displace functions
                for (func_name, func) in displace_function
                    init_time = time()
                    func(pwldr)
                    end_time = time()
                    
                    pwldr_model = NFPModelPWLDR(problem, pwldr)

                    reoptm_pwldr_optimize = nfp_second_stage(pwldr_model, samples_demand_test, samples_cost_test)
                    dr_pwldr_optimize = dr_pwldr_calc(pwldr_model.model, samples_test)
                    obj_value_pwldr_optimize = pwldr_model.objective_value

                    push!(results_df, (
                            dist = dist_name,
                            idx_problem = idx,
                            n_segments = seg,
                            displace_func = func_name,
                            reoptm_standard_form = reoptm_standard_form,
                            reoptm_ldr = reoptm_ldr,
                            reoptm_pwldr_uniform = reoptm_pwldr_uniform,
                            reoptm_pwldr_optimize = reoptm_pwldr_optimize,
                            reoptm_eev = reoptm_eev,
                            dr_ldr = dr_ldr,
                            dr_pwldr_uniform = dr_pwldr_uniform,
                            dr_pwldr_optimize = dr_pwldr_optimize,
                            obj_value_ldr = obj_value_ldr,
                            obj_value_pwldr_uniform = obj_value_pwldr_uniform,
                            obj_value_pwldr_optimize = obj_value_pwldr_optimize,
                            obj_value_eev = obj_value_eev,
                            obj_value_standard_form = obj_value_standard_form,
                            ws = ws,
                            opt_time = end_time - init_time
                        ), promote = true)
                    
                    println("Checkpoint: dist=$dist_name, func=$func_name, n_segments=$seg")
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

optimizer = HiGHS.Optimizer
network_flow_planning_test(dist_list, optimizer, 200, 2000, 3)
shipment_planning_test(dist_list, optimizer, 200, 2000, 3)
