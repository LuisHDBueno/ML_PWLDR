function _evaluate_local_search(
    pwldr_model::PWLDR,
    sense::Int
)
    optimize!(pwldr_model) 
    return sense * objective_value(pwldr_model)
end

function _local_search_η_vec!(
    pwldr_model::PWLDR,
    weight_vec::Vector{Vector{Float64}},
    index::Int;
)
    if pwldr_model.model.ext[:sense] == MOI.MIN_SENSE
        sense = 1
    else
        sense = -1
    end

    value = _evaluate_local_search(pwldr_model, sense)

    size = length(weight_vec[index])
    weight = weight_vec[index]

    improved = true
    while improved
        
        improved = false

        for i in 2:(size)
            weight_copy = copy(weight)
            step = 1/(length(weight) * 10)
            if (weight[i] - step > 0)
                weight_copy[i] = weight[i] - step
                weight_vec[index] = weight_copy
                update_breakpoints!(pwldr_model, weight_vec)
                value_lower = _evaluate_local_search(pwldr_model, sense)
                if value_lower < value
                    improved = true
                    value = value_lower
                    weight = weight_copy
                    break
                end
            end

            if (weight[i] + step <= 1)
                weight_copy[i] = weight[i] + step
                weight_vec[index] = weight_copy

                update_breakpoints!(pwldr_model, weight_vec)
                value_upper = _evaluate_local_search(pwldr_model, sense)
                if value_upper < value
                    improved = true
                    value = value_upper
                    weight = weight_copy
                    break
                end
            end

            weight_vec[index] = weight
        end
    end
    update_breakpoints!(pwldr_model, weight_vec)
end

function local_search_independent!(
    pwldr_model::PWLDR
)
    weight_vec = Vector{Vector{Float64}}()
    for pwvr in pwldr_model.PWVR_list
        push!(weight_vec, pwvr.weight)
    end

    for i in 1:length(pwldr_model.PWVR_list)
        _local_search_η_vec!(pwldr_model, weight_vec, i)
        println("otimizado $i")
    end
end