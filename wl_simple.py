import numpy as np


def wl_simple(graph_db, h=4):
    # Create one empty feature vector for each graph
    feature_vectors = []
    offset = 0
    graph_indices = []
    for g in graph_db:
        feature_vectors.append(np.zeros(0, dtype=np.float64))
        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    colors = []
    for g in graph_db:
        g.vp.l = g.new_vertex_property("int")
        for v in g.vertices():
            colors.append(hash(g.vp.nl[v]))
            g.vp.l[v] = hash(g.vp.nl[v])

    _, colors = np.unique(colors, return_inverse=True)
    max = int(np.amax(colors) + 1)

    for g in graph_db:
        feature_vectors = [
            np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1] + 1], minlength=max))) for
            i, index in enumerate(graph_indices)]

    for _ in range(h):
        colors = []
        for g in graph_db:
            for v in g.vertices():
                neighbors = []

                for n in v.all_neighbours():
                    neighbors.append(g.vp.l[n])

                neighbors.sort()
                neighbors.append(g.vp.l[v])
                colors.append(hash(tuple(neighbors)))

        _, colors = np.unique(colors, return_inverse=True)

        q = 0
        for g in graph_db:
            for v in g.vertices():
                g.vp.l[v] = colors[q]
                q += 1

        max = int(np.amax(colors) + 1)

        feature_vectors = [
            np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1] + 1], minlength=max))) for
            i, index in enumerate(graph_indices)]

    return np.array(feature_vectors)