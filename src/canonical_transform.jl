
function _build_B(
    B::SparseArrays.SparseMatrixCSC{Float64, Int64},
    η_min::Vector{Float64},
    n_segments::Vector{I}
    ) where I

    n_cols = Int(sum(n_segments)) + 1
    n_rows = size(B, 1)
    B_new = zeros(n_rows, n_cols)
    B_new[:,1] = B[:,1]
    col = 2

    for i in 1:length(η_min)
        B_new[:, 1] += B[:, i + 1] * η_min[i]
        for _ in 1:n_segments[i]
            B_new[:, col] = B[:, i + 1]
            col += 1
        end
    end

    return B_new
end

function _build_C(
    C::SparseArrays.SparseMatrixCSC{Float64, Int64},
    η_min::Vector{Float64},
    n_segments::Vector{I}
    ) where I
    return _build_B(C, η_min, n_segments)
end

function _build_h(
    n_segments::Vector{F}
) where F
    h = zeros(Int(sum(n_segments)) + 1 + 2)
    # "1" in (1, n)
    h[1] = 1
    h[2] = -1

    col = 2
    # eta inequalities
    for i in n_segments
        h[col + 1] = -1
        col += i
    end

    return h
end

function _build_W(
    n_segments::Vector{F},
    PWVR_list::Vector{PWVR}
) where F
    W = zeros(Int(sum(n_segments)) + 1 + 2,
                Int(sum(n_segments)) + 1)
                
    # "1" in (1, n)
    W[1,1] = 1
    W[2,1] = -1

    line = 3
    for pwvr in PWVR_list
        η_vec = pwvr.η_vec
        for i in 2:length(η_vec)
            diff = η_vec[i] - η_vec[i-1]
            W[line, line - 1] = -1/diff
            W[line + 1, line - 1] = 1/diff
            line += 1
        end
    end

    return W
end