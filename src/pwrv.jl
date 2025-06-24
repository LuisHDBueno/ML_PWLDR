
mutable struct PWVR
    distribution
    n_breakpoints::Int64
    min::Float64
    max::Float64
    range::Float64
    weight::Vector{Float64}
    η_vec::Vector{Float64}
end

function PWVR(
    distribution,
    min,
    max,
    weight
)
    range = max - min
    n_breakpoints = length(weight)
    η_vec = [min]
    for i in 1:n_breakpoints
        push!(η_vec, last(η_vec) + range * weight[i])
    end

    return PWVR(distribution, n_breakpoints, min, max, range, weight, η_vec)
end

function update_breakpoints!(variable::PWVR, new_weight)
    #Normalizar os pesos para somar 1 e limitar superiormente η
    w_norm = new_weight ./ sum(new_weight)
    variable.weight = w_norm
    for i in 1:variable.n_breakpoints
        variable.η_vec[i + 1] = variable.η_vec[i] + variable.range * variable.weight[i]
    end
end

function mean(variable::PWVR)
    return Distributions.mean(variable.distribution) - variable.min
end

function mean(variable::PWVR, segment_index)
    min = variable.η_vec[segment_index - 1]
    max = variable.η_vec[segment_index]
    trunc_variable = truncated(variable.distribution, min, max)
    return Distributions.mean(trunc_variable) - min
end

function var(variable::PWVR)
    return Distributions.var(variable.distribution)
end

function var(variable::PWVR, segment_index)
    min = variable.η_vec[segment_index - 1]
    max = variable.η_vec[segment_index]
    return Distributions.var(truncated(variable.distribution, min, max))
end
