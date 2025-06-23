function n_out_indices(arc_list, node)
    return [i for (i, (m,n)) in enumerate(arc_list) if m == node]
end

function n_in_indices(arc_list, node)
    return [i for (i, (m,n)) in enumerate(arc_list) if n == node]
end

function shortest_path(n_nodes::Int, n_edges::Int, ns::Int, nt::Int,
                         distribution, optimizer)
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    possible_arcs = [(i, j) for i in 1:n_nodes, j in 1:n_nodes if i != j]
    A = unique(rand(sample(possible_arcs, length(possible_arcs); replace=false), n_edges))

    @variable(ldr, flow[1:n_edges])

    dists = [distribution for _ in 1:n_edges]
    joint_dist = Product(dists)
    @variable(ldr, ξ[1:n_edges] in LinearDecisionRules.Uncertainty(
        distribution = joint_dist
    ))

    # Out
    @constraint(ldr, sum(flow[i] for i in n_out_indices(A, ns)) == 1)

    # In
    @constraint(ldr, sum(flow[i] for i in n_in_indices(A, nt)) == -1)

    for n in 1:n_nodes
        if n != ns && n != nt
            @constraint(ldr,
                sum(flow[i] for i in n_out_indices(A, n)) ==
                sum(flow[i] for i in n_in_indices(A, n))
            )
        end
    end

    @objective(ldr, Max, -sum(ξ[i] * flow[i] for i in 1:n_edges))

    return ldr
end