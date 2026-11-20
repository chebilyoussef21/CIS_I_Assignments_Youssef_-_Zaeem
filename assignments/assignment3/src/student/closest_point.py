"""Closest-point queries against the triangle mesh."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import vct3
from ..mesh import Mesh

def convert_to_N_by_3_array(points) -> np.ndarray:
    #helper function created by us
    # in this function we convert a list of vct3 or (3,) array into an (N,3) float array.
    #used in register_points()
    point_list = []
    for point in points:
        if isinstance(point, vct3):
            point_list.append(point.vec.ravel())
        else:
            point_list.append(point)
    return np.array(point_list, dtype=float)

def closest_point_on_triangle(p: vct3, a: vct3, b: vct3, c: vct3) -> vct3:
    """
    input: query point p, triangle corners a, b, c
    output: the point of the triangle (face, edge or vertex) closest to p
    """
    a, b, c, p = convert_to_N_by_3_array([a, b, c, p])

    plane = np.cross(b - a, c - a)
    numerator = np.dot(p - a, plane)
    denominator = np.linalg.norm(plane) ** 2
    projection = p - (numerator / denominator) * plane
    edge_1_triangle = (b-a)
    edge_2_triangle = (c-a)
    # weights = np.linalg.solve(np.array([edge_1_triangle, edge_2_triangle]).T, projection - a)
    weights = np.linalg.lstsq(np.array([edge_1_triangle, edge_2_triangle]).T, projection - a, rcond=None)[0]
    weight_a = 1 - weights[0] - weights[1]
    vertices = np.array([a, b, c])
    total_weights = np.array([weight_a, weights[0], weights[1]])
    if np.all(total_weights >= 0):
        return vct3(projection)
    elif np.any(total_weights < 0,axis=0):
        non_negative_weights = total_weights[total_weights >= 0]
        if len(non_negative_weights)==1:
            # idx = np.flatnonzero(total_weights >= non_negative_weights[0])[0]
            # closest_vertex = vertices[idx]
            closest_point = closest_point_on_triangle_edges(p, a[None], b[None], c[None])[0]
            return vct3(closest_point)
        if len(non_negative_weights) ==2:
            # idx = np.flatnonzero(total_weights < 0)
            idx = np.flatnonzero(total_weights >= 0)
            edge_vertices = vertices[idx]
            edge = (edge_vertices[1] - edge_vertices[0])
            edge_length = np.dot(edge, edge)
            t = np.clip(np.dot(projection - edge_vertices[0], edge) / edge_length, 0, 1)
            closest_point = edge_vertices[0] + t * edge
            return vct3(closest_point)
        raise ValueError("No physical explanation")
    
def closest_point_on_triangle_edges(point: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    #Nearest point to `point` on the boundary (3 edges) of each triangle.
    best_points, best_sq_dist = None, None
    for start, end in ((A, B), (B, C), (C, A)):
        edge = end - start
        t = np.clip(np.einsum('ij,ij->i', point - start, edge) /
                    np.maximum(np.einsum('ij,ij->i', edge, edge), 1e-300), 0, 1)
        edge_points = start + t[:, None] * edge
        sq_dist = np.einsum('ij,ij->i', edge_points - point, edge_points - point)
        if best_points is None:
            best_points, best_sq_dist = edge_points, sq_dist
        else:
            closer = sq_dist < best_sq_dist
            best_points[closer], best_sq_dist[closer] = edge_points[closer], sq_dist[closer]
    return best_points


def closest_points_on_mesh(points: np.ndarray, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Closest point on the mesh for every query point (vectorised over triangles).
    input: points (K,3) in CT coordinates, mesh
    output: closest (K,3), triangle index (K,), distance (K,)
    """
    triangle_vertices = mesh.tri_vertices 
    A, B, C = triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2]
    AB, AC = B - A, C - A
    d00 = np.einsum('ij,ij->i', AB, AB)
    d01 = np.einsum('ij,ij->i', AB, AC)
    d11 = np.einsum('ij,ij->i', AC, AC)
    denom = d00 * d11 - d01 ** 2

    num_points = points.shape[0]
    closest_points = np.zeros((num_points, 3))
    closest_triangle_indices = np.zeros(num_points, dtype=int)
    closest_distances = np.zeros(num_points)

    for point_idx, point in enumerate(points):
        AP = point - A
        d20 = np.einsum('ij,ij->i', AP, AB)
        d21 = np.einsum('ij,ij->i', AP, AC)
        with np.errstate(divide='ignore', invalid='ignore'):
            weight_b = (d11 * d20 - d01 * d21) / denom
            weight_c = (d00 * d21 - d01 * d20) / denom
        weight_a = 1 - weight_b - weight_c
        candidate_points = A + weight_b[:, None] * AB + weight_c[:, None] * AC   # projection onto each plane

        outside = ~((weight_a >= 0) & (weight_b >= 0) & (weight_c >= 0))
        if outside.any():
            candidate_points[outside] = closest_point_on_triangle_edges(point, A[outside], B[outside], C[outside])

        candidate_distances = np.linalg.norm(candidate_points - point, axis=1)
        closest_index = np.argmin(candidate_distances)
        closest_points[point_idx] = candidate_points[closest_index]
        closest_triangle_indices[point_idx] = closest_index
        closest_distances[point_idx] = candidate_distances[closest_index]

    return closest_points, closest_triangle_indices, closest_distances


# def closest_points_on_mesh(points: np.ndarray, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     implement (vectorised; a per-triangle Python loop is far too slow).
#     input: points (K,3) in CT coordinates, mesh
#     output: closest (K,3), triangle index (K,), distance (K,)
#     """
#     ct_triangle_vertices = mesh.tri_vertices
#     num_points = points.shape[0]

#     closest_points = np.zeros((num_points, 3))           # closest mesh point for each query point
#     closest_triangle_indices = np.zeros(num_points, dtype=int)  # which triangle it lies on
#     closest_distances = np.zeros(num_points)             # distance from query point to that mesh point

#     #for point in points:
#     for point_idx, point in enumerate(points):
#         #closest_points = np.array([closest_point_on_triangle(point, ct_triangle_vertices[i][0], ct_triangle_vertices[i][1], ct_triangle_vertices[i][2]) for i in range(ct_triangle_vertices.shape[0])]) 
#         #possible answers for one query point -> each triangle has 1 candidate point
#         candidate_points = np.array([
#             closest_point_on_triangle(point, ct_triangle_vertices[i][0], ct_triangle_vertices[i][1], ct_triangle_vertices[i][2])
#             for i in range(ct_triangle_vertices.shape[0])
#         ])
#         candidate_distances = np.linalg.norm(candidate_points - point, axis=1)        
#         closest_index = np.argmin(candidate_distances)

#         closest_points[point_idx] = candidate_points[closest_index]
#         closest_triangle_indices[point_idx] = closest_index
#         closest_distances[point_idx] = candidate_distances[closest_index]

#     # return closest_points[closest_index], closest_index, distances[closest_index]
#     return closest_points, closest_triangle_indices, closest_distances