
function _build_second_moment_matrix(
    n_segments::Vector{Float64},
    η_vec_list::Vector{Any}
)

    n_cols = Int(sum(n_segments)) + length(n_segments)
    M = zeros(n_cols, n_cols)

    #TODO: Change to any distribution, fixed to uniform
    M[1,1] = 1

    line = 2
    for η_vec in η_vec_list
        for i in 2:length(η_vec)
            #Uniform distribution mean
            expec_per_interval = (η_vec[i] - η_vec[i - 1])/2
            M[line, 1] = expec_per_interval
            M[1, line] = expec_per_interval

            # i = j
            M[line, line] = ((η_vec[i] - η_vec[i-1])^3) / 3
            line += 1
        end
    end

    line = 1
    # i < j
    for η_vec in η_vec_list
        init_line_block = line + 1
        last_line_block = length(η_vec) + init_line_block - 1
        Δ = (η_vec[length(η_vec)] - η_vec[1])
        for i in (init_line_block):(last_line_block)
            init_col_block = i + 1
            last_col_block = last_line_block - 1
            for j in (init_col_block):(last_col_block)
                Δ_j = (η_vec[i] - η_vec[i-1])
                Δ_i = (η_vec[i] - η_vec[i-1])
                value_i_j =  Δ_i * Δ_j^2 / (2 * Δ)
                value_j_max = Δ_i * Δ_j * (η_vec[length(η_vec)] - η_vec[j]) / Δ
                value = value_i_j + value_j_max
                M[i, j] = value
                M[j, i] = value
            end
            line += 1
        end
    end
    
    return M
end
