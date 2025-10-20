"""
    _segments_number(
        ldr_model::LinearDecisionRules.LDRModel;
        fix_n::Int = 1
    )

    Build a vector with fixed segments number for each random variable of the
        ldr model

    # Arguments
    ldr_model::LinearDecisionRules.LDRModel: Original LDR problem
    fix_n::Int = 1: Number to be fixed

    # Returns
    ::Vector{Int}: Respective segments_number vector
"""
function _segments_number(
    ldr_model::LinearDecisionRules.LDRModel;
    fix_n::Int = 1
)
    ABC = ldr_model.ext[:_LDR_ABC]
    dim_ξ_ldr = size(ABC.Be, 2)

    n_segments_vec = ones(Int, dim_ξ_ldr - 1) * fix_n
    return n_segments_vec
end

"""
    set_breakpoint!(
        pwldr::PWLDR,
        variable::JuMP.VariableRef,
        n_breakpoints::Int
    )

    Set the number of breakpoints for the respective variable

    # Arguments
    - pwldr::PWLDR: PWLDR model
    - variable::JuMP.VariableRef: Name of the variable to be changed
    - n_breakpoints::Int: Number of breakpoints to be setted
"""
function set_breakpoint!(
    pwldr::PWLDR,
    variable::JuMP.VariableRef,
    n_breakpoints::Int
)
    dist_idx, inner_idx = pwldr.uncertainty_to_distribution[variable]
    pwldr.n_segments_vec[dist_idx] = n_breakpoints + 1
    pwldr.reset_model = true
end