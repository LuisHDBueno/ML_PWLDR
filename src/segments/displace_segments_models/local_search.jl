function _evaluate_local_search(PWVR_list, ABC, first_stage_index, n_segments_vec, optimizer)

    model = _build_problem(ABC, first_stage_index, PWVR_list, n_segments_vec, optimizer)
    optimize!(model)
    
    values = objective_value(model)
    
    return values
end

function _local_search_η_vec(η_vec, index, PWVR_list, ABC, first_stage_index, n_segments_vec, optimizer; tolerance = 2)
    improved = true
    while improved

        improved = false
        for i in 2:(length(η_vec) - 1)
            
            value_η = _evaluate_local_search(PWVR_list, ABC, first_stage_index, n_segments_vec, optimizer)

            η_vec_copy = copy(η_vec)
            if (η_vec[i] - η_vec[i - 1] >= tolerance)
                lower = (η_vec[i] + η_vec[i - 1])/2
                η_vec_copy[i] = lower
                PWVR_list[index].η_vec = η_vec_copy

                value_η_copy = _evaluate_local_search(PWVR_list, ABC, first_stage_index, n_segments_vec, optimizer)
                if value_η_copy > value_η
                    η_vec = η_vec_copy
                    improved = true
                    break
                end
            end
            if (η_vec[i + 1] - η_vec[i] >= tolerance)
                upper = (η_vec[i] + η_vec[i + 1])/2
                η_vec_copy[i] = upper
                PWVR_list[index].η_vec = η_vec_copy

                value_η_copy = _evaluate_local_search(PWVR_list, ABC, first_stage_index, n_segments_vec, optimizer)
                if value_η_copy > value_η
                    η_vec = η_vec_copy
                    improved = true
                    break
                end
            end
        end
    end

    return η_vec
end

function local_search_independent(
    η_min,
    η_max,
    ABC,
    first_stage_index,
    n_segments_vec,
    optimizer,
    distribution_constructor
)

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
    for (i, pwvr) in enumerate(PWVR_list)
        values = _local_search_η_vec(pwvr.η_vec, i, PWVR_list,
                                         ABC, first_stage_index, n_segments_vec,
                                         optimizer)
        pwvr.η_vec = values
        println("otimizado $i")
    end

    return PWVR_list
end

function local_search_greed(
    η_min,
    η_max,
    ABC,
    first_stage_index,
    n_segments_vec,
    optimizer,
    distribution_constructor
)
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
    list_copy = copy(PWVR_list)

    while !isempty(list_copy)
        best_value = -Inf
        best_i = 0
        best_η = nothing
        best_pwvr = nothing

        for (i, pwvr) in enumerate(list_copy)
            η_vec = _local_search_η_vec(pwvr.η_vec, i, PWVR_list,
                                         ABC, first_stage_index, n_segments_vec,
                                         optimizer)
            value = evaluate(pwvr, η_vec)

            if value > best_value
                best_value = value
                best_i = i
                best_η = η_vec
                best_pwvr = pwvr
            end
        end
        deleteat!(list_copy, best_i)

        for pwvr in PWVR_list
            if pwvr === best_pwvr
                pwvr.η_vec = best_η
                break
            end
        end
    end
    return PWVR_list
end
