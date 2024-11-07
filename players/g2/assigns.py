from shapely.geometry import Polygon
import math
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import numpy as np
import miniball


"""def filter_valid_polygons(polygons: list[Polygon], radius=12.5) -> list[Polygon]:
    removed_indices = []
    for i, polygon in enumerate(polygons):
        if not (isinstance(polygon, Polygon) and can_cake_fit_in_plate(polygon, radius)):
            removed_indices.append(i)

    print(f"Removed indices: {removed_indices}")
    print(f"Total polygons removed: {len(removed_indices)}")

    return [polygon for i, polygon in enumerate(polygons) if i not in removed_indices]
"""


def can_cake_fit_in_plate(cake_piece, radius=12.5):
    if not isinstance(cake_piece, Polygon):
        raise TypeError("Expected a Polygon object.")
    cake_points = np.array(list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double)
    res = miniball.miniball(cake_points)

    return res["radius"] <= radius


def calculate_total_penalty(
    assignment: list[int], polygons: list[Polygon], requests: list[float], d: float
) -> float:
    total_penalty = 0.0

    for request_idx, polygon_idx in enumerate(assignment):
        if polygon_idx == -1:
            # Skip unmatched requests (no penalty for unmatched requests assumed)
            continue

        request_size = requests[request_idx]
        if request_size == 0:
            continue  # Avoid division by zero for dummy requests

        polygon_area = polygons[polygon_idx].area

        # Check if the polygon fits on the plate
        if not can_cake_fit_in_plate(polygons[polygon_idx]):
            # Full penalty for invalid polygons
            total_penalty += 100  # Assuming maximum penalty for not fitting
            continue

        # Calculate the percentage difference from the requested size
        area_diff_percentage = abs(polygon_area - request_size) / request_size * 100

        # If the percentage difference exceeds the tolerance, add the penalty
        if area_diff_percentage > d:
            total_penalty += area_diff_percentage

    return total_penalty


def assign(polygons: list[Polygon], requests: list[float], d: float) -> list[int]:
    # Filter valid polygons
    valid_polygons = polygons

    # List of assignment function names
    assignment_functions = [
        "hungarian_min_penalty",
        "greedy_best_fit_assignment",
        "dp_min_penalty",
        "sorted_assign",
    ]

    # Generate assignments and their penalties
    assignments = []
    penalties = []

    for func_name in assignment_functions:
        # Call the assignment function by name
        assignment_func = globals()[func_name]
        assignment = assignment_func(valid_polygons, requests, d)
        penalty = calculate_total_penalty(assignment, valid_polygons, requests, d)

        assignments.append(assignment)
        penalties.append(penalty)

    # Find the assignment with the lowest penalty
    min_penalty_index = penalties.index(min(penalties))
    print(f"The best method is: {assignment_functions[min_penalty_index]}")
    print(f"Here is the total penalty: {min(penalties)}")
    return assignments[min_penalty_index]


def sorted_assign(
    polygons: list[Polygon], requests: list[float], d: float
) -> list[int]:
    # Get sorted indices of polygons and requests in decreasing order of area
    sorted_polygon_indices = sorted(
        range(len(polygons)), key=lambda i: polygons[i].area, reverse=True
    )
    sorted_request_indices = sorted(
        range(len(requests)), key=lambda i: requests[i], reverse=True
    )

    # Assign each sorted polygon to each sorted request by index
    assignment = [-1] * len(requests)

    for i in range(min(len(sorted_polygon_indices), len(sorted_request_indices))):
        polygon_idx = sorted_polygon_indices[i]
        request_idx = sorted_request_indices[i]
        assignment[request_idx] = polygon_idx  # Match request index to polygon index

    return assignment


def greedy_best_fit_assignment(
    polygons: list[Polygon], requests: list[float], d: float
) -> list[int]:
    # sort requests and polygons in descending order by size
    sorted_polygons = sorted(enumerate(polygons), key=lambda x: x[1].area, reverse=True)
    sorted_requests = sorted(enumerate(requests), key=lambda x: x[1], reverse=True)

    assignment = [-1] * len(requests)
    used_polygons = set()

    for request_idx, request_size in sorted_requests:
        best_fit_polygon_idx = None
        min_penalty = float("inf")
        closest_area_diff = float("inf")

        for poly_idx, polygon in sorted_polygons:
            if poly_idx in used_polygons:
                continue

            # Check if the polygon fits on the plate
            if not can_cake_fit_in_plate(polygon):
                penalty = 100  # Full penalty for polygons that do not fit
            else:
                area_diff_percentage = (
                    abs(polygon.area - request_size) / request_size * 100
                )
                penalty = area_diff_percentage if area_diff_percentage > d else 0

            # find the polygon with minimum penalty and closest area to the request
            if (penalty < min_penalty) or (
                penalty == min_penalty
                and abs(polygon.area - request_size) < closest_area_diff
            ):
                best_fit_polygon_idx = poly_idx
                min_penalty = penalty
                closest_area_diff = abs(polygon.area - request_size)

        if best_fit_polygon_idx is not None:
            assignment[request_idx] = best_fit_polygon_idx
            used_polygons.add(best_fit_polygon_idx)

    return assignment


def dp_min_penalty(
    polygons: list[Polygon], requests: list[float], d: float
) -> list[int]:
    n = len(polygons)
    m = len(requests)
    dp = [[math.inf] * (m + 1) for _ in range(n + 1)]
    assignment = [-1] * m

    dp[0][0] = 0

    # this fills the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            area_diff = (
                abs(polygons[i - 1].area - requests[j - 1]) / requests[j - 1] * 100
            )
            penalty = area_diff if area_diff > d else 0

            dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1] + penalty)

    # backtrack for assignment
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i][j] == dp[i - 1][j - 1] + (
            abs(polygons[i - 1].area - requests[j - 1]) / requests[j - 1] * 100
            if abs(polygons[i - 1].area - requests[j - 1]) / requests[j - 1] * 100 > d
            else 0
        ):
            assignment[j - 1] = i - 1
            i -= 1
            j -= 1
        else:
            i -= 1

    return assignment


def hungarian_min_penalty(
    polygons: list[Polygon], requests: list[float], d: float
) -> list[int]:
    # Define a zero area polygon as one with all coordinates (0, 0)
    def is_zero_area_polygon(polygon: Polygon) -> bool:
        return all(coord == (0, 0) for coord in polygon.exterior.coords)

    # Copy polygons and requests
    polygons_copy = polygons[:]
    requests_copy = requests[:]

    num_polygons = len(polygons_copy)
    num_requests = len(requests_copy)

    # Add dummy requests or dummy polygons as needed
    if num_polygons > num_requests:
        num_dummy_requests = num_polygons - num_requests
        requests_copy += [0] * num_dummy_requests  # Add dummy requests with no penalty
        print(f"Added {num_dummy_requests} dummy requests.")
    elif num_requests > num_polygons:
        num_dummy_polygons = num_requests - num_polygons
        polygons_copy += [
            Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
        ] * num_dummy_polygons  # Add zero area polygons to inflict full penalty
        print(f"Added {num_dummy_polygons} dummy polygons.")

    # Build cost matrix (penalties)
    cost_matrix = []
    for request_size in requests_copy:
        row = []
        for polygon in polygons_copy:
            # Check if the polygon fits on the plate
            if not can_cake_fit_in_plate(polygon):
                penalty = 100  # Full penalty if polygon doesn't fit
            else:
                polygon_area = (
                    polygon.area if not is_zero_area_polygon(polygon) else 0
                )  # Handle zero area polygons
                if request_size == 0:  # No penalty for dummy requests
                    penalty = 0
                else:
                    area_diff = abs(polygon_area - request_size) / request_size * 100
                    penalty = area_diff if area_diff > d else 0
            row.append(penalty)
        cost_matrix.append(row)

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Prepare the final assignment, filtering out dummy requests and polygons
    assignment = [-1] * num_requests
    for i in range(len(row_ind)):
        if row_ind[i] < num_requests and col_ind[i] < num_polygons:
            assignment[int(row_ind[i])] = int(col_ind[i])

    # Verify data type and format
    assert isinstance(assignment, list), "Assignment should be a list"
    assert all(
        isinstance(x, int) for x in assignment
    ), "All elements in assignment should be integers"
    assert len(assignment) == len(
        requests
    ), "Assignment length should match the number of requests"
    assert all(
        0 <= x < len(polygons) for x in assignment if x != -1
    ), "Indices in assignment should refer to valid polygons"

    # Print for debugging
    print("Assignment:", assignment)
    print("Type of assignment:", type(assignment))

    # Return a copy to prevent unexpected modifications
    return assignment[:]
