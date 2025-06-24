
function _build_second_moment_matrix(
    n_segments::Vector{F},
    PWVR_list::Vector{PWVR}
) where F

    n_cols = Int(sum(n_segments)) + 1
    M = zeros(n_cols, n_cols)

    M[1,1] = 1

    line = 2
    for pwvr in PWVR_list
        η_vec = pwvr.η_vec
        for i in 2:length(η_vec)
            expec_per_interval = mean(pwvr, i)
            M[line, 1] = expec_per_interval
            M[1, line] = expec_per_interval

            # i = j
            mean_shift = expec_per_interval - η_vec[i - 1]
            variance = var(pwvr, i)
            prob_segment = cdf(pwvr.distribution, η_vec[i]) - cdf(pwvr.distribution, η_vec[i - 1])
            M[line, line] = (mean_shift + variance) * prob_segment
            line += 1
        end
    end

    line = 1
    # i < j
    for pwvr in PWVR_list
        η_vec = pwvr.η_vec

        init_line_block = line + 1
        last_line_block = length(η_vec) + init_line_block - 2

        for i_matrix in (init_line_block):(last_line_block)
            init_col_block = i_matrix + 1
            last_col_block = last_line_block - 2

            i = i_matrix - init_line_block + 2

            for j_matrix in (init_col_block):(last_col_block)   
                j = j_matrix - init_col_block + 2

                Δ_j = (η_vec[j] - η_vec[j - 1])
                Δ_i = (η_vec[i] - η_vec[i - 1])

                # η_j-1 < x < η_j
                mean_shift = mean(pwvr, j) - η_vec[j-1]
                p_interval = cdf(pwvr.distribution, η_vec[j]) - cdf(pwvr.distribution, η_vec[j - 1])
                value_1 = Δ_i * mean_shift * p_interval

                # η_j < x < η_max
                p_tail = 1 - cdf(pwvr.distribution, η_vec[j])
                value_2 = Δ_i * Δ_j * p_tail

                value = value_1 + value_2
                M[i_matrix, j_matrix] = value
                M[j_matrix, i_matrix] = value
            end
            line += 1
        end
    end
    
    return M
end
