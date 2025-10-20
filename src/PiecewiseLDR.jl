module PiecewiseLDR
    using Distributions
    using Expectations
    using JuMP
    using Statistics
    using Random
    using LinearDecisionRules
    using LinearAlgebra
    using SparseArrays
    using BlackBoxOptim
    import JuMP: optimize!, objective_value, getindex

    include("pwrv.jl")
    include("canonical_transform.jl")
    include("second_moment.jl")
    include("pwldr.jl")
    include("segments/opt_segments.jl")

end #End module