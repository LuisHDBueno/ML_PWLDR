
function _segments_number(ldr_model; fix_n = 3)
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