import Distributions: cdf

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
    n_breakpoints = length(weight) - 1
    η_vec = [min]
    for i in 1:(n_breakpoints + 1)
        push!(η_vec, last(η_vec) + range * weight[i])
    end
    η_vec[end] = max

    return PWVR(distribution, n_breakpoints, min, max, range, weight, η_vec)
end

function sample_vector(variable::PWVR, value::Float64)
    ξ_tilde = zeros(variable.n_breakpoints + 1)
    ξ_tilde[1] = variable.min
    value -= variable.min
    idx = 1
    while value > 0
        Δi = Δ(variable, idx)
        ξ_tilde[idx] += min(Δi, value)
        value -= Δi
        idx += 1
    end
    return ξ_tilde
end

function update_breakpoints!(variable::PWVR, new_weight::Vector{Float64})
    #Normalizar os pesos para somar 1 e limitar superiormente η
    w_norm = new_weight ./ sum(new_weight)
    variable.weight = w_norm
    for i in 1:variable.n_breakpoints
        variable.η_vec[i + 1] = variable.η_vec[i] + variable.range * variable.weight[i]
    end
end

function mean(variable::PWVR)
    return [mean(variable, i) for i in 1:(variable.n_breakpoints+1)]
end

function mean(variable::PWVR, segment_index)
    min = variable.η_vec[segment_index]
    max = variable.η_vec[segment_index + 1]

    trunc_dist = truncated(variable.distribution, min, max)
    prob = prob_between(variable, min, max)
    E = Expectations.expectation(trunc_dist)
    exp = E(x -> x - min)
    Δi = Δ(variable, segment_index)
    val = exp * prob + Δi * (1 - Distributions.cdf(variable.distribution, max))
    if (segment_index == 1)
        val += variable.min
    end
    return val
end

function var(variable::PWVR)
    return [var(variable, i) for i in 1:(variable.n_breakpoints+1)]
end

function var(variable::PWVR, segment_index)
    Δi = Δ(variable, segment_index)

    min = variable.η_vec[segment_index]
    max = variable.η_vec[segment_index + 1]
    trunc_dist = truncated(variable.distribution, min, max)

    E = Expectations.expectation(trunc_dist)

    prob1 = prob_between(variable, min, max)
    part1 = E(x -> (x - min)^2) * prob1

    part2 = Δi^2 * (1 - Distributions.cdf(variable.distribution, max))

    μ = mean(variable, segment_index)
    if (segment_index == 1)
        μ -= variable.min
    end
    return part1 + part2 - μ^2
end

function cov(variable::PWVR)
    n = variable.n_breakpoints + 1
    μ = mean(variable)
    μ[1] -= variable.min
    ret = zeros(Float64, n, n)

    for j in 1:n
        for i in 1:(j-1)
            Δi = Δ(variable, i)
            Δj = Δ(variable, j)

            min_j = variable.η_vec[j]
            max_j = variable.η_vec[j + 1]
            trunc_dist_j = truncated(variable.distribution, min_j, max_j)
            prob_j = prob_between(variable, min_j, max_j)

            E = Expectations.expectation(trunc_dist_j)
            part1 = Δi * (E(x -> x - min_j) * prob_j)
            if j == 1
                part1 += variable.min * prob_j
            end

            part2 = Δi * Δj * (1 - Distributions.cdf(variable.distribution, max_j))
            val = part1 + part2 - μ[i] * μ[j]

            ret[i, j] = val
            ret[j, i] = val

        end
    end

    # diagonal = variâncias
    v = var(variable)
    for i in 1:n
        ret[i, i] = v[i]
    end

    return ret
end

function cov(pwvr_i::PWVR, pwvr_j::PWVR)
    Σ = zeros(pwvr_i.n_breakpoints + 1, pwvr_j.n_breakpoints + 1)
    for i in 1:(pwvr_i.n_breakpoints + 1)
        for j in 1:(pwvr_j.n_breakpoints + 1)
            Σ[i,j] = cov(pwvr_i, i, pwvr_j, j)
        end
    end
    return Σ
end

function cov(pwvr_i::PWVR, idx_i::Int, pwvr_j::PWVR, idx_j::Int)
    dist_i = truncated(pwvr_i.distribution, pwvr_i.η_vec[idx_i], pwvr_i.η_vec[idx_i + 1])
    dist_j = truncated(pwvr_j.distribution, pwvr_j.η_vec[idx_j], pwvr_j.η_vec[idx_j + 1])
    return cov(dist_i, dist_j)
end 

function cov(dist1::Distribution, dist2::Distribution; n_samples=1_000)
    rng = MersenneTwister(1234)
    x = rand(rng, dist1, n_samples)
    y = rand(rng, dist2, n_samples)
    μx = sum(x)/n_samples
    μy = sum(y)/n_samples

    Σ = sum((x .- μx) .* (y .- μy))/n_samples
    return Σ
end

function prob_between(variable::PWVR, min, max)
    return cdf(variable.distribution, max) - cdf(variable.distribution, min)
end

function Δ(variable::PWVR, segment_index)
    return variable.η_vec[segment_index + 1] - variable.η_vec[segment_index]
end