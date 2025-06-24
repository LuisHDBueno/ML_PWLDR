function evaluate_segments(weights, PWVR_list, ABC, first_stage_index, n_segments_vec, optimizer)

    weight_index = 1
    for i in 1:length(PWVR_list)
        w = weights[weight_index:weight_index + Int(n_segments_vec[i] - 1)]
        update_breakpoints!(PWVR_list[i], w)
        weight_index += n_segments_vec[i]
    end

    model = _build_problem(ABC, first_stage_index, PWVR_list, n_segments_vec, optimizer)
    optimize!(model)
    
    values = objective_value(model)
    
    return values
end

function black_box(
    η_min,
    η_max,
    ABC,
    first_stage_index,
    n_segments_vec,
    optimizer,
    distribution_constructor
)
    η_weights_list = Vector{Tuple{Float64, Float64}}()
    for i in 1:length(η_min)
        for _ in 1:n_segments_vec[i]
            # Pesos de cada um dos limites
            push!(η_weights_list, (0.001, 1.0))
        end
    end

    PWVR_list = Vector{PWVR}()
    for i in 1:length(η_min)
        n = n_segments_vec[i]
        push!(PWVR_list,
                PWVR(distribution_constructor(η_min[i], η_max[i]),
                        η_min[i],
                        η_max[i],
                        fill(1/n, Int(n)))
                )
    end

    obj_func = hyperparam -> evaluate_segments(hyperparam, PWVR_list,
                                                ABC, first_stage_index,
                                                n_segments_vec, optimizer)
    bboptimize(obj_func, SearchRange = η_weights_list, NumDimensions = length(η_weights_list), MaxFuncEvals = 100)

    return PWVR_list
end
