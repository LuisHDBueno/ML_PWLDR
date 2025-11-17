function open_params()
    filepath = joinpath(@__DIR__, "params.json") 
    if !isfile(filepath)
        error("Arquivo de parâmetros não encontrado: $filepath")
    end
    
    json_data = JSON.parsefile(filepath)
    return Vector{Float64}(json_data)
end

function set_opt_breakpoint_number!(
    pwldr::PWLDR,
    variable::JuMP.VariableRef;
    α::Float64 = 0.05,
    max_bp::Int = 10,
    n_samples::Int = 100
)
    β = open_params()
    V = vector_representation(pwldr, variable; n_samples)
    nb = floor(
        (β[1] + sum(β[2:16] .* V) + α) / (β[17] + sum(β[18:32] .* V))
    )
    if max_bp > nb
        nb = max_bp
    end
    set_breakpoint!(pwldr, variable, nb)
end