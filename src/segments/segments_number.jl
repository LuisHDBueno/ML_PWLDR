
function _segments_number(ldr_model; fix_n = 1)
    ABC = ldr_model.ext[:_LDR_ABC]
    dim_ξ_ldr = size(ABC.Be, 2)

    n_segments_vec = zeros(Int, dim_ξ_ldr - 1)
    dim_ξ = 1
    for i in 1:(dim_ξ_ldr - 1)
        dim_ξ += fix_n
        n_segments_vec[i] = fix_n
    end
    return n_segments_vec
end

function set_breakpoint!(pwldr, variable, n_breakpoints)
    dist_idx, inner_idx = pwldr.uncertainty_to_distribution[variable]
    pwldr.n_segments_vec[dist_idx] = n_breakpoints + 1
end