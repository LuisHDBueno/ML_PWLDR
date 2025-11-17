""""
    ProblemSetup

    Struct that hold the functions to build instances of a problem in different
    stochastic setups

    # Fields
    - name: String to identify the type of problem
    - gen_metadata: Function that generate random metadata for a probleminstance
    - second_stage: Function to optimze the second stage given a first stage
        decision
    - ldr: LDR setup
    - ws: Wait-and-See setup
    - std: Standard Form setup
    - deterministc: Deterministc setup
"""
struct ProblemSetup
    name
    gen_metadata
    second_stage
    ldr
    ws
    std
    deterministic
end

"""
    ProblemInstance

    Struct to hold a instance of a problem given a stochastic setup

    # Fields
    - metadata: Data to build the model
    - first_stage_decision: First stage decision of the model
    - model: Stochastic setup
    - objective_value: Objective value of the model
    - test_value: Value after reoptimize second-stage
"""
mutable struct ProblemInstance
    metadata
    first_stage_decision
    model
    objective_value
    test_value
end
