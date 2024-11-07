from typing import List, Tuple, Callable
from sklearn.mixture import GaussianMixture
import numpy as np

import logging
import constants
from shapely.geometry import Polygon

import miniball

def get_polygon_areas(polygons):
    """Calculates and returns sorted polygon indices based on their area."""
    polygons_with_area = [(i, polygon.area) for i, polygon in enumerate(polygons)]
    sorted_polygons = sorted(polygons_with_area, key=lambda x: x[1])
    return sorted_polygons  # Each tuple is (index, area)

def create_ratio_groups(sorted_requests, group_size):
    """Creates groups from sorted_requests where each group maintains a consistent ratio progression."""
    groups = [[] for _ in range(group_size)]
    for i, element in enumerate(sorted_requests):
        group_index = i % group_size
        groups[group_index].append(element)
    transposed_groups = [list(group) for group in zip(*groups)]
    return transposed_groups

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, tolerance: int) -> None:
        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None

    def can_cake_fit_in_plate(self, cake_piece, radius=12.5):
        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def _calculate_penalty(
        self, assign_func: Callable[[List[Polygon], List[float]], List[int]]
    ) -> float:
        penalty = 0
        assignments: List[int] = assign_func(self.polygons, self.requests)

        for request_index, assignment in enumerate(assignments):
            # Check if the cake piece fits on a plate of diameter 25
            if assignment == -1 or not self.can_cake_fit_in_plate(self.polygons[assignment]):
                penalty += 100
            else:
                # Calculate penalty based on the difference in area
                penalty_percentage = (
                    100
                    * abs(self.polygons[assignment].area - self.requests[request_index])
                    / self.requests[request_index]
                )
                # Add penalty if it exceeds tolerance
                if penalty_percentage > self.tolerance:
                    penalty += penalty_percentage
        return penalty

    def determine_optimal_horizontal_cuts(self, groups, max_cuts=3) -> int:
        """Determine the optimal number of horizontal cuts based on group ratios."""
        best_num_cuts = 0
        best_ratio_variance = float('inf')

        # Calculate group sum ratios to compare against horizontal cuts
        group_sums = [sum(group) for group in groups]
        total_sum = sum(group_sums)
        normalized_ratios = [group_sum / total_sum for group_sum in group_sums]

        # Test each possible number of horizontal cuts
        for num_cuts in range(1, max_cuts + 1):
            # Calculate the expected segment ratios for `num_cuts` horizontal cuts
            segment_ratios = [(i + 1) / (num_cuts + 1) for i in range(num_cuts)]
            
            # Measure the variance in ratio alignment with group sums
            ratio_variance = np.var([abs(seg - norm) for seg, norm in zip(segment_ratios, normalized_ratios[:num_cuts])])
            
            # Update if this number of cuts has the lowest variance
            if ratio_variance < best_ratio_variance:
                best_ratio_variance = ratio_variance
                best_num_cuts = num_cuts

        return best_num_cuts
    
    def get_ratios(self, groups):
        group_sums = [sum(group) for group in groups]
        total_sum = sum(group_sums)
        normalized_ratios = [group_sum / total_sum for group_sum in group_sums]
        return normalized_ratios
    
    def simulate_subgame(self, requests: List[float]) -> Tuple[int, List[List[float]], float]:
        """Simulate the entire subgame, pre-determining the best path of cuts based on penalty."""
        best_cut_config = None
        best_penalty = float('inf')
        best_cut_coords = []

        # Iterate over possible numbers of cuts (0 to 3 for this example)
        for i in range(4):
            # Generate vertical cuts and group requests for ratio consistency
            vertical_cuts, groups = self.generate_vertical_cuts(requests, num_cuts=i)
            ratios = self.get_ratios(groups)

            # Inject crumbs and horizontal cuts
            cut_coords = inject_crumb_coords(vertical_cuts, self.cake_len, self.cake_width)
            cut_coords = inject_horizontal_cuts(ratios, cut_coords, self.cake_len, self.cake_width, i)

            # Compute assignment and penalty for this configuration
            penalty = self._calculate_penalty(lambda polys, reqs: self.assign_pieces(reqs, polys))

            print(f"Configuration {i}: Cut Coords: {cut_coords}, Penalty: {penalty}")

            # Update best configuration if this one has a lower penalty
            # if penalty < best_penalty:
            #     best_penalty = penalty
            #     best_cut_config = (i, cut_coords)

            if i == 3:
                best_penalty = penalty
                best_cut_config = (i, cut_coords)


        # Store the best configuration's cuts and penalty for retrieval in `move`
        best_num_hor_cuts, best_cut_coords = best_cut_config
        self.best_cut_coords = best_cut_coords
        self.best_penalty = best_penalty
        print(f"Optimal configuration selected: Cuts: {best_cut_coords}, Penalty: {best_penalty}")

        return best_num_hor_cuts, best_cut_coords, best_penalty

    def move(self, current_percept) -> Tuple[int, List[int]]:
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width

        self.polygons = polygons
        self.requests = requests

        print('Turn', turn_number, '\n')

        # Run the subgame simulation only on the first turn
        if turn_number == 1:
            _, self.best_cut_coords, self.best_penalty = self.simulate_subgame(requests)

        # Use the pre-determined best path of cuts on each turn
        if turn_number == 1:
            return constants.INIT, self.best_cut_coords[0]

        if turn_number < len(self.best_cut_coords) + 1:
            return constants.CUT, self.best_cut_coords[turn_number - 1]
        
        # Assign pieces after all cuts are completed
        assignment = self.assign_pieces(requests, polygons)
        return constants.ASSIGN, assignment
    
    def assign_pieces(self, requests: List[float], polygons: List[float]) -> List[int]:
        """Assign each request to the polygon with the closest area, minimizing the sum of differences."""
        assignment = [-1] * len(requests)
        
        requests_sorted = sorted(enumerate(requests), key=lambda x: x[1])
        polygons_sorted = get_polygon_areas(polygons)
        
        j = 0 
        for req_idx, req_size in requests_sorted:
            best_poly_idx = -1
            best_diff = float('inf')
            
            for k in range(j, len(polygons_sorted)):
                poly_idx, poly_size = polygons_sorted[k]
                diff = abs(poly_size - req_size)
                
                if diff < best_diff:
                    best_diff = diff
                    best_poly_idx = k
                
                if poly_size > req_size:
                    break

            if best_poly_idx != -1:
                poly_idx, poly_size = polygons_sorted.pop(best_poly_idx)
                assignment[req_idx] = poly_idx
                j = max(0, j - 1)

        self.polygons = polygons

        return assignment

    def generate_vertical_cuts(self, requests, num_cuts=0):
        """Generates vertical cuts with consistent ratio grouping for requests."""
        
        requests_copy = requests[:]
        group_size = num_cuts + 1
        fakes = []

        # Ensure the number of requests is a multiple of the group size
        if len(requests_copy) % group_size != 0:
            required_requests = group_size - (len(requests_copy) % group_size)
            
            # Add "fake" requests based on the median to keep distribution consistent
            gmm = GaussianMixture(n_components=min(2, len(requests_copy)), random_state=0)
            gmm.fit(np.array(requests_copy).reshape(-1, 1))
            smallest_cluster_mean = gmm.means_.flatten()[np.argmin(np.unique(gmm.predict(np.array(requests_copy).reshape(-1, 1)), return_counts=True)[1])]
            requests_copy.extend([smallest_cluster_mean] * required_requests)

        # Sort requests and create groups to maintain consistent ratios
        sorted_requests = sorted(requests_copy)
        groups = create_ratio_groups(sorted_requests, group_size)

        # Generate x-widths for vertical cuts based on adjusted groups
        x_widths = [sum(group) / self.cake_len for group in groups]
        x_coords = [round(sum(x_widths[:i+1]), 2) for i in range(len(x_widths))]

        # Format x and y coordinates for vertical cuts
        x_coords = [x for coord in x_coords for x in [coord, coord]]
        y_coords = [0, self.cake_len, self.cake_len, 0]
        vertical_cuts = [[x, y] for x, y in zip(x_coords, y_coords * (len(x_coords) // 4))]

        return vertical_cuts, groups

def inject_crumb_coords(vertical_cuts, cake_len, cake_width):
    final_cuts = []

    for i, cut in enumerate(vertical_cuts):
        final_cuts.append(cut)

        if (i + 1) % 2 == 0:
            final_cuts.append(get_crumb_coord(cut, cake_len, cake_width))

    return final_cuts

def get_crumb_coord(cut, cake_len, cake_width):
    x = cake_width if cut[0] > (cake_width / 2) else 0
    knife_error = 0.01
    y = round(cake_len - knife_error, 2) if cut[1] == cake_len else knife_error
    return [x, y]

def inject_horizontal_cuts(ratios, vertical_cuts, cake_len, cake_width, num_hor_cuts):
    """Injects a specified number of horizontal cuts into the cake."""
    horizontal_cuts = []
    total_y = 0
    for i in range(num_hor_cuts):
        y = round(total_y + cake_len * ratios[i], 2)
        total_y += cake_len * ratios[i]
        print(total_y)
        
        if i % 2 == 0:
            horizontal_cuts.extend([[0, y], [cake_width, y]])
            if i + 1 < num_hor_cuts:
                horizontal_cuts.extend([[cake_width, y], [cake_width, round(total_y + cake_len * ratios[i+1], 2)]])
        else:
            horizontal_cuts.extend([[cake_width, y], [0, y]])
            if i + 1 < num_hor_cuts:
                horizontal_cuts.extend([[0, y], [0, round(total_y + cake_len * ratios[i+1], 2)]])

    horizontal_cuts = inject_hor_crumb_coords(horizontal_cuts, cake_len, cake_width)

    return horizontal_cuts + vertical_cuts

def inject_hor_crumb_coords(horizontal_cuts, cake_len, cake_width):
    final_cuts = []

    for i, cut in enumerate(horizontal_cuts):
        final_cuts.append(cut)

        if (i + 1) % 2 == 0:
            final_cuts.append(get_hor_crumb_coord(cut, cake_len, cake_width))

    return final_cuts

def get_hor_crumb_coord(cut, cake_len, cake_width):
    y = cake_len if cut[1] > (cake_len / 2) else 0
    knife_error = 0.01
    x = round(cake_width - knife_error, 2) if cut[0] == cake_width else knife_error

    return [x, y]