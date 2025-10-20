import Distributions: cdf

"""
    PWVR

    Mutable struct that represents a piecewise linear variable with lower and
        upper bounds

    # Fields
    - distribution::Distributions.Distribution: Probability distribution
    - n_breakpoints::Int64: Number of breakpoints to split the variable
    - min::Float64: Lower bound of the variable's support
    - max::Float64: Upper bound of the variable's support
    - range::Float64: Total range of the variable (`max - min`)
    - weight::Vector{Float64}: Weight associated with each segment of the
        picewise variable
    - η_vec::Vector{Float64}: Vector containing the nodes values used in the
         piecewise representation. Includes the min and max values.
"""
mutable struct PWVR
    distribution::Distributions.Distribution
    n_breakpoints::Int64
    min::Float64
    max::Float64
    range::Float64
    weight::Vector{Float64}
    η_vec::Vector{Float64}
end

"""
    PWVR(
        distribution,
        min,
        max,
        weight
    )

    Contructor for the PWVR structure

    # Arguments
    - distribution::Distributions.Distribution: Probability distribution
    - n_breakpoints::Int64: Number of breakpoints to split the variable
    - min::Float64: Lower bound of the variable's support
    - max::Float64: Upper bound of the variable's support
    - range::Float64: Total range of the variable (`max - min`)
    - weight::Vector{Float64}: Weight associated with each segment of the
        picewise variable
    - η_vec::Vector{Float64}: Vector containing the nodes values used in the
         piecewise representation. Includes the min and max values.

    # Returns
    ::PWVR: Mutable struct that represents a piecewise linear variable with
        lower and upper bounds
"""
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

"""
    sample_vector(
        variable::PWVR,
        value::Float64
    )

    Build the lifted vector of the piecewise variable at given sample

    # Arguments
    - variable::PWVR: Piecewise variable to be lifted
    - value::Float64: Value of the sample

    # Errors
    - ArgumentError: Thrown if `value` is outside the valid range `[variable.min, variable.max]`.

    # Returns
    ::Vector{Float64}: Lifted vector
"""
function sample_vector(
    variable::PWVR,
    value::Float64
)
    if !(variable.min <= value <= variable.max)
        throw(ArgumentError("Value $value is out of bounds:
                            [$(variable.min), $(variable.max)]"))
    end

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

"""
    update_breakpoints!(
        variable::PWVR,
        new_weight::Vector{Float64}
    )

    Normalize the vector of weights for each segment and update the breakpoints
        of the piecewise variable

    # Arguments
    - variable::PWVR: Piecewise variable to be updated
    - new_weight::Vector{Float64}: Vector of weights for each segment
"""
function update_breakpoints!(
    variable::PWVR,
    new_weight::Vector{Float64}
)
    w_norm = new_weight ./ sum(new_weight)
    variable.weight = w_norm
    for i in 1:variable.n_breakpoints
        variable.η_vec[i + 1] = variable.η_vec[i] + variable.range * variable.weight[i]
    end
end

"""
    mean(
        variable::PWVR
    )

    Build a vector with the mean of each segment for the piecewise variable

    # Arguments
    - variable::PWVR: Piecewise variable to get lifted mean

    # Return
    ::Vector{Float64}: Lifted mean vector
"""
function mean(
    variable::PWVR
)
    return [mean(variable, i) for i in 1:(variable.n_breakpoints+1)]
end

"""
    mean(
        variable::PWVR,
        segment_index::Int
    )
    
    Get the mean of the random variable at the respective segment

    # Arguments
    - variable::PWVR: Piecewise variable to get mean
    - segment_index::Int: Number of the segment to get the mean

    # Return
    - ::Float64: Mean of the segment
"""
function mean(
    variable::PWVR,
    segment_index::Int
)
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

"""
    var(
        variable::PWVR
    )

    Build a vector with the variance of each segment for the piecewise variable

    # Arguments
    - variable::PWVR: Piecewise variable to get the lifted variance

    # Return
    ::Vector{Float64}: Lifted variance vector
"""
function var(
    variable::PWVR
)
    return [var(variable, i) for i in 1:(variable.n_breakpoints+1)]
end

"""
    var(
        variable::PWVR,
        segment_index::Int
    )
    
    Get the var of the random variable at the respective segment

    # Arguments
    - variable::PWVR: Piecewise variable to get the variance
    - segment_index::Int: Number of the segment to get the variance

    # Returns
    ::Float64: Variance of the segment
"""
function var(
    variable::PWVR,
    segment_index::Int
)
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

"""
    cov(
        variable::PWVR
    )

    Get the covariance matrix between the random variables that represents the
        piecewise variable

    # Arguments
    - variable::PWVR: Piecewise variable to get the covariance

    # Returns
    ::Matrix{Float64}: Covariance Matrix
"""
function cov(
    variable::PWVR
)
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

"""
    cov(
        pwvr_i::PWVR,
        pwvr_j::PWVR
    )

    Get the covariance matrix between each segment of two independent piecewise
        random variables

    # Arguments
        - pwvr_i::PWVR: First piecewise random variable
        - pwvr_j::PWVR: Second piecewise random variable

    # Returns
    ::Matrix{Float64}: Covariance Matrix
"""
function cov(
    pwvr_i::PWVR,
    pwvr_j::PWVR
)
    Σ = zeros(pwvr_i.n_breakpoints + 1, pwvr_j.n_breakpoints + 1)
    for i in 1:(pwvr_i.n_breakpoints + 1)
        for j in 1:(pwvr_j.n_breakpoints + 1)
            Σ[i,j] = cov(pwvr_i, i, pwvr_j, j)
        end
    end
    return Σ
end

"""
    cov(
        pwvr_i::PWVR,
        idx_i::Int,
        pwvr_j::PWVR,
        idx_j::Int
    )

    Get the covariance between segments of two independent piecewise random
        variables

    # Arguments
        - pwvr_i::PWVR: First piecewise random variable
        - idx_i::Int: Index of the segment at the first random variable
        - pwvr_j::PWVR: Second piecewise random variable
        - idx_j::Int: Index of the segment at the second random variable

    # Returns
    ::Float64: Covariance between segments
"""
function cov(
    pwvr_i::PWVR,
    idx_i::Int,
    pwvr_j::PWVR,
    idx_j::Int
)
    dist_i = truncated(pwvr_i.distribution, pwvr_i.η_vec[idx_i], pwvr_i.η_vec[idx_i + 1])
    dist_j = truncated(pwvr_j.distribution, pwvr_j.η_vec[idx_j], pwvr_j.η_vec[idx_j + 1])
    return cov(dist_i, dist_j)
end 

"""
    cov(
        dist1::Distribution,
        dist2::Distribution;
        n_samples = 1_000
    )

    Get the covariance between two distributions using n_samples

    # Arguments
    - dist1::Distribution: First distribution
    - dist2::Distribution: Second distribution
    - n_samples = 1_000: Number of samples to calculate the samples

    # Returns
    ::Float64: Covariance between distributions
"""
function cov(
    dist1::Distribution,
    dist2::Distribution;
    n_samples = 1_000
)
    rng = MersenneTwister(1234)
    x = rand(rng, dist1, n_samples)
    y = rand(rng, dist2, n_samples)
    μx = sum(x)/n_samples
    μy = sum(y)/n_samples

    Σ = sum((x .- μx) .* (y .- μy))/n_samples
    return Σ
end

"""
    prob_between(
        variable::PWVR,
        min::Float64,
        max::Float64
    )
    
    Get the probability between two values at a piecewise random variable

    # Arguments
    - variable::PWVR: Piecewise variable to be considered
    - min::Float64: Lower value of the range
    - max::Float64: Upper value of the range
"""
function prob_between(
    variable::PWVR,
    min::Float64,
    max::Float64
)
    return cdf(variable.distribution, max) - cdf(variable.distribution, min)
end

"""
    Δ(
        variable::PWVR,
        segment_index::Int
    )
    
    Get the size of the support at the segment on the piecewise variable

    # Arguments
    - variable::PWVR: Piecewise variable to get support
    - segment_index::Int: Index of the segment to be considered

    # Returns
    ::Float64: Size of the support at the segment
"""
function Δ(
    variable::PWVR,
    segment_index::Int
)
    return variable.η_vec[segment_index + 1] - variable.η_vec[segment_index]
end