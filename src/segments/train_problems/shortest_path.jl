function n_out_indices(arc_list, node)
    return [i for (i, (m,n)) in enumerate(arc_list) if m == node]
end

function n_in_indices(arc_list, node)
    return [i for (i, (m,n)) in enumerate(arc_list) if n == node]
end

function generate_connected_random_graph(n_nodes::Int, n_edges::Int)

    shuffled_nodes = shuffle(2:n_nodes-1)
    path_nodes = [1; shuffled_nodes; n_nodes]
    path_edges = [(path_nodes[i], path_nodes[i+1]) for i in 1:length(path_nodes)-1]

    possible_arcs = [(i, j) for i in 1:n_nodes, j in 1:n_nodes if i != j && (i, j) ∉ path_edges]
    n_extra_edges = n_edges - length(path_edges)

    extra_edges = sample(possible_arcs, n_extra_edges; replace=false)

    A = vcat(path_edges, extra_edges)

    return A
end

function shortest_path(n_nodes::Int, n_edges::Int, ns::Int, nt::Int,
                         distribution, optimizer)
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    A = generate_connected_random_graph(n_nodes, n_edges)

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

    @objective(ldr, Min, sum(ξ[i] * flow[i] for i in 1:n_edges))

    return ldr, A
end

function shortest_path_PI(A, ξ, ns::Int, nt::Int, optimizer)
    ldr = LinearDecisionRules.LDRModel(optimizer)
    set_silent(ldr)

    @variable(ldr, flow[1:n_edges])

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

    @objective(ldr, Min, sum(ξ[i] * flow[i] for i in 1:n_edges))

    optimize!(ldr)

    return objective_value(ldr)
end