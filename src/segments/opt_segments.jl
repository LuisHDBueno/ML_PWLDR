function _evaluate_segments(
    weights_limits_list::Vector{Float64},
    pwldr_model::PWLDR)

    weight_vec = Vector{Vector{Float64}}()
    weight_index = 1
    for i in 1:length(pwldr_model.PWVR_list)
        push!(weight_vec,
            weights_limits_list[weight_index:weight_index + Int(pwldr_model.n_segments_vec[i] - 1)]
            )
        weight_index += pwldr_model.n_segments_vec[i]
    end

    update_breakpoints!(pwldr_model, weight_vec)
    optimize!(pwldr_model)
    return objective_value(pwldr_model)
end

function opt_segments!(
    pwldr_model::PWLDR
)
    η_weights_list = Vector{Tuple{Float64, Float64}}()
    for n in pwldr_model.n_segments_vec
        for _ in 1:n
            # Pesos de cada um dos limites
            push!(η_weights_list, (0.001, 1.0))
        end
    end
    obj_func = hyperparam -> _evaluate_segments(hyperparam, pwldr_model)
    bboptimize(obj_func, SearchRange = η_weights_list, NumDimensions = length(η_weights_list), MaxFuncEvals = 100)
end
