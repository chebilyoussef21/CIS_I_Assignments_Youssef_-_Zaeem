"""Closest-point queries against the triangle mesh."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import vct3
from ..mesh import Mesh


def closest_point_on_triangle(p: vct3, a: vct3, b: vct3, c: vct3) -> vct3:
    """
    TODO: implement.
    input: query point p, triangle corners a, b, c
    output: the point of the triangle (face, edge or vertex) closest to p
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    p = np.asarray(p)
    plane = np.cross(b - a, c - a)
    numerator = np.dot(p - a, plane)
    denominator = np.linalg.norm(plane) ** 2
    projection = p - (numerator / denominator) * plane
    edge_1_triangle = (b-a)
    edge_2_triangle = (c-a)
    weights = np.linalg.solve(np.array([edge_1_triangle, edge_2_triangle]).T, projection - a)
    weight_a = 1 - weights[0] - weights[1]
    vertices = np.array([a, b, c])
    total_weights = np.array([weight_a, weights[0], weights[1]])
    if np.all(total_weights >= 0):
        return projection
    elif np.any(total_weights < 0,axis=0):
        non_negative_weights = total_weights[total_weights >= 0]
        if len(non_negative_weights)==1:
            idx = np.flatnonzero(total_weights == non_negative_weights[0])[0]
            closest_vertex = vertices[idx]
            return closest_vertex
        if len(non_negative_weights) ==2:
            idx = np.flatnonzero(total_weights < 0)
            edge_vertices = vertices[idx]
            edge = (edge_vertices[1] - edge_vertices[0])
            edge_length = np.dot(edge, edge)
            t = np.clip(np.dot(projection - edge_vertices[0], edge) / edge_length, 0, 1)
            closest_point = edge_vertices[0] + t * edge
            return closest_point
        return ValueError("No physical explanation")
    


def closest_points_on_mesh(points: np.ndarray, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: implement (vectorised; a per-triangle Python loop is far too slow).
    input: points (K,3) in CT coordinates, mesh
    output: closest (K,3), triangle index (K,), distance (K,)
    """
    raise NotImplementedError("TODO: implement closest_points_on_mesh")
