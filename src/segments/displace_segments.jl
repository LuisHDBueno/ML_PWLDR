function build_η_vec_list(η_min, η_max, n_segments_vec, weights)
    η_vec_list = []
    weight_index = 1
    for i in 1:length(η_min)
        total_length = η_max[i] - η_min[i]

        w = weights[weight_index:weight_index + Int(n_segments_vec[i] - 1)]
        weight_index += n_segments_vec[i]

        #Normalizar os pesos para somar 1 e limitar superiormente η
        w_norm = w ./ sum(w)

        #Comeca do minimo e soma ate o maximo com pesos distintos
        η_vec = [η_min[i]]
        for j in 1:length(w_norm)
            push!(η_vec, last(η_vec) + total_length * w_norm[j])
        end
        push!(η_vec_list, η_vec)
    end

    return η_vec_list
end

function evaluate_segments(weights, η_min, η_max, ABC, first_stage_index, n_segments_vec, optimizer)

    η_vec_list = build_η_vec_list(η_min, η_max, n_segments_vec, weights)

    model = _build_problem(ABC, first_stage_index, η_vec_list, n_segments_vec, optimizer)
    optimize!(model)
    
    values = -objective_value(model)
    
    return values
end

function _displace_segments(
    η_min,
    η_max,
    ABC,
    first_stage_index,
    n_segments_vec,
    optimizer
)
    η_weights_list = Vector{Tuple{Float64, Float64}}()
    for i in 1:length(η_min)
        for _ in 1:n_segments_vec[i]
            # Pesos de cada um dos limites
            push!(η_weights_list, (0.001, 1.0))
        end
    end

    obj_func = hyperparam -> evaluate_segments(hyperparam, η_min, η_max,
                                                ABC, first_stage_index,
                                                n_segments_vec, optimizer)
    opt_result =  bboptimize(obj_func, SearchRange = η_weights_list, NumDimensions = length(η_weights_list), MaxFuncEvals = 100)

    best_η_weights = best_candidate(opt_result)
    best_η_vec_list = build_η_vec_list(η_min, η_max, n_segments_vec, best_η_weights)

    return best_η_vec_list
end
