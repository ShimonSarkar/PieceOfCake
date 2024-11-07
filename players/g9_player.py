import bisect
from itertools import accumulate
import math
from typing import List

import numpy as np
from sympy import divisors
import logging
import pulp

import constants


class Player:
    def __init__(
            self,
            rng: np.random.Generator,
            logger: logging.Logger,
            precomp_dir: str,
            tolerance: int,
    ) -> None:
        """Initialise the player with the basic information

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("message")
            precomp_dir (str): Directory path to store/load pre-computation
            tolerance (int): tolerance for the cake distribution
            cake_len (int): Length of the smaller side of the cake
        """

        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None

    def move(self, current_percept) -> tuple[int, List[int]]:
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width
        noise = 0

        print(" ")
        print(
            f"----------------------------------- Turn {turn_number} -----------------------------------"
        )

        # First turn initialize knife to start at first vertical cut
        if turn_number == 1:
            self.cut_coords = compute_cuts(
                requests, cake_len, cake_width, cake_area, noise, self.tolerance
            )
            print(f"Cut coordinates {len(self.cut_coords)}: {self.cut_coords}")
            return constants.INIT, self.cut_coords[turn_number - 1]

        # Cut the cake
        if turn_number < len(self.cut_coords) + 1:
            return constants.CUT, self.cut_coords[turn_number - 1]

        # Semi Naive Assignment - closest cake piece starting from largest request (works for now since all cuts have a specific request in mind)
        sorted_requests = sorted(
            [(requests[i], i) for i in range(len(requests))], reverse=True
        )
        polygons_available = {(polygons[i].area, i) for i in range(len(polygons))}
        assignment = [-1] * len(requests)

        for req, i in sorted_requests:
            closest_index = min(polygons_available, key=lambda x: abs(x[0] - req))[1]
            polygons_available.remove((polygons[closest_index].area, closest_index))
            assignment[i] = closest_index

        return constants.ASSIGN, assignment


def test_best_grid_cuts(requests, cake_len, cake_width, cake_area, tolerance):

    # Bounds based on the size of cake that fits on a plate
    min_horz_cuts = math.ceil(cake_len / 24.6)
    max_horz_cuts = math.floor(math.sqrt(len(requests)))

    # Get best grid cut with original list of requests
    altered_piece = None
    best_num_horz_cuts, best_x_coords, min_total_penalty = get_best_grid_cuts(
        requests, cake_len, cake_width, cake_area, min_horz_cuts, max_horz_cuts, tolerance, altered_piece
    )

    # Test if adding fake request pieces minimizes penalty to increase factors in cake cuts.
    for r in set(requests):  # Use a set to avoid duplicate processing
        altered_piece = r
        altered_requests = requests + [altered_piece]
        num_horz_cuts, x_coords, curr_penalty = get_best_grid_cuts(
            altered_requests, cake_len, cake_width, cake_area, min_horz_cuts, max_horz_cuts, tolerance, altered_piece
        )

        if curr_penalty is not None and (min_total_penalty is None or curr_penalty <= min_total_penalty):
            min_total_penalty = curr_penalty
            best_x_coords = x_coords
            best_num_horz_cuts = num_horz_cuts

    return best_num_horz_cuts, best_x_coords, min_total_penalty



def compute_cuts(requests, cake_len, cake_width, cake_area, noise, tolerance):
    num_horz_cuts, x_coords, total_penalty = test_best_grid_cuts(
        requests,
        cake_len,
        cake_width,
        cake_area,
        tolerance,
    )

    vertical_cut_coords = get_vertical_cuts(
        requests, cake_len, cake_width, cake_area, noise, x_coords
    )
    cut_coords = inject_crumb_coords(vertical_cut_coords, cake_len, cake_width, True)

    if cut_coords[-1][1] == 0:
        horizontal_cut_coords = get_horizontal_cuts(
            num_horz_cuts, cake_len, cake_width, False
        )
        cut_coords.append([round(cake_width - 0.01, 2), 0])
    else:
        horizontal_cut_coords = get_horizontal_cuts(
            num_horz_cuts, cake_len, cake_width, True
        )
        cut_coords.append([round(cake_width - 0.01, 2), cake_len])

    horz_cut_coords = inject_crumb_coords(
        horizontal_cut_coords, cake_len, cake_width, False
    )
    cut_coords += horz_cut_coords

    return cut_coords


def get_vertical_cuts(requests, cake_len, cake_width, cake_area, noise, x_coords):
    # Duplicate the x coordinates because a single vertical cut will require two x coordinates.
    x_coords = [x for x in x_coords for _ in range(2)]

    # Establish the cutting pattern of down then up (over 4 coordinates) which will repeat.
    y_coords = [0, cake_len, cake_len, 0]
    # Combine x and y coordinates to create tuples. two sets of consecutive coordinates represents a single vertical cut. These are already ordered.
    vertical_cuts = [
        [x, y] for x, y in zip(x_coords, y_coords * math.ceil(len(x_coords) / 4))
    ]

    return vertical_cuts


def inject_crumb_coords(cuts, cake_len, cake_width, is_vertical):
    complete_cut_coords = []
    for i, coord in enumerate(cuts):
        complete_cut_coords.append(coord)
        # Add an extra element after every second string
        if (i + 1) % 2 == 0:
            complete_cut_coords.append(
                get_crumb_coord(coord, cake_len, cake_width, is_vertical)
            )
    return complete_cut_coords


def get_crumb_coord(xy_coord, cake_len, cake_width, is_vertical):
    # Set y based on if we are currently at the top or bottom of the cake
    knife_error = 0.01

    if is_vertical:
        # Set x based on which half of the cake we are in
        crumb_x = cake_width if xy_coord[0] > (cake_width / 2) else 0

        crumb_y = (
            round(cake_len - knife_error, 2) if xy_coord[1] == cake_len else knife_error
        )
    else:
        crumb_x = (
            round(cake_width - knife_error, 2)
            if xy_coord[0] == cake_width
            else knife_error
        )
        crumb_y = cake_len if xy_coord[1] > (cake_len / 2) else 0

    return [round(crumb_x, 2), round(crumb_y, 2)]


def get_horizontal_cuts(num_horz_cuts, cake_len, cake_width, ends_at_bottom):
    x_coords = [cake_width, 0, 0, cake_width]
    cut_coords = list(accumulate([cake_len / num_horz_cuts] * (num_horz_cuts - 1)))
    if ends_at_bottom:
        cut_coords = cut_coords[::-1]
    y_coords = [y for y in [round(n, 2) for n in cut_coords] for _ in range(2)]
    horizontal_cuts = [
        [x, y] for x, y in zip(x_coords * math.ceil(len(y_coords) / 4), y_coords)
    ]
    return horizontal_cuts


# num_horz_cuts truly means horizontal areas but we can pretend that we would need to make a cut along the top of the cake.
# Would rather have it be this way so that it is consistent with num_vert_cuts (we do need to make a final cut for the extra 5%)
def get_best_grid_cuts(
        requests, cake_len, cake_width, cake_area, min_horz_cuts, max_horz_cuts, tolerance, altered_piece
):
    requests_sorted = sorted(requests, reverse=True)

    # Adjusted tolerance for rounding errors
    tolerance = max(1, tolerance - 1)
    t = tolerance / 100

    # Compute possible numbers of horizontal cuts (rows)
    requests_factors = divisors(len(requests))
    l_i = bisect.bisect_left(requests_factors, min_horz_cuts)
    r_i = bisect.bisect_right(requests_factors, max_horz_cuts)
    possible_horz_cuts = requests_factors[l_i:r_i]

    min_total_penalty = float("inf")
    best_num_horz_cuts = None
    best_vert_cut_diffs = None

    for num_horz_cuts in possible_horz_cuts:
        num_vert_cuts = len(requests) // num_horz_cuts
        horz_cut_diff = cake_len / num_horz_cuts

        h = horz_cut_diff  # Height of each piece

        # Initialize MILP problem
        prob = pulp.LpProblem("Minimize_Penalty", pulp.LpMinimize)

        # Variables
        x_int = pulp.LpVariable.dicts(
            "x_int", (i for i in range(num_vert_cuts)), lowBound=1, cat="Integer"
        )
        x = {i: 0.01 * x_int[i] for i in range(num_vert_cuts)}

        s_j = pulp.LpVariable.dicts(
            "s_j", (j for j in range(len(requests))), lowBound=0, cat="Continuous"
        )

        y_j = pulp.LpVariable.dicts(
            "y_j", (j for j in range(len(requests))), cat="Binary"
        )

        delta_j = pulp.LpVariable.dicts(
            "delta_j", (j for j in range(len(requests))), lowBound=0, cat="Continuous"
        )

        # Objective Function
        alt_index = requests.index(altered_piece) if altered_piece is not None else -1
        prob += pulp.lpSum([s_j[j] for j in range(len(requests)) if j != alt_index])

        # Total Width Constraint (Adjusted)
        prob += pulp.lpSum([x[i] for i in range(num_vert_cuts)]) <= cake_width

        M = 1e6  # A large number for big-M method

        # Constraints
        request_idx = 0
        for i in range(num_vert_cuts):
            curr_requests = requests_sorted[i * num_horz_cuts: (i + 1) * num_horz_cuts]
            A_i = h * x[i]

            for r in curr_requests:
                r_value = r

                # Tolerance Constraints
                prob += A_i >= (1 - t) * r_value - M * y_j[request_idx]
                prob += A_i <= (1 + t) * r_value + M * y_j[request_idx]

                # Deviation Constraints (Only active when y_j == 1)
                prob += delta_j[request_idx] >= A_i - r_value - M * (
                        1 - y_j[request_idx]
                )
                prob += delta_j[request_idx] >= r_value - A_i - M * (
                        1 - y_j[request_idx]
                )
                prob += delta_j[request_idx] >= 0

                # Penalty Constraints
                beta_j = 100 / r_value  # Precompute constant
                prob += s_j[request_idx] >= delta_j[request_idx] * beta_j
                prob += s_j[request_idx] <= M * y_j[request_idx]

                request_idx += 1

        # Solve the MILP
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Check if a feasible solution was found
        if prob.status != pulp.LpStatusOptimal:
            continue  # Try the next number of horizontal cuts

        # Extract the solution
        vert_cut_diffs = [pulp.value(x[i]) for i in range(num_vert_cuts)]
        total_penalty = sum(pulp.value(s_j[j]) for j in range(len(requests)))

        print(
            f"Number of Horizontal Cuts: {num_horz_cuts}, Total Penalty: {total_penalty}"
        )

        if total_penalty <= min_total_penalty:
            min_total_penalty = total_penalty
            best_num_horz_cuts = num_horz_cuts
            best_vert_cut_diffs = vert_cut_diffs.copy()

    if best_num_horz_cuts is None:
        print("No feasible solution found.")
        return None, None, None

    print("MIN PENALTY", min_total_penalty)
    print("BEST VERT", best_vert_cut_diffs)
    accum_vert_cuts = list(accumulate(best_vert_cut_diffs))
    accum_vert_cuts = [round(diff, 2) for diff in accum_vert_cuts]
    return best_num_horz_cuts, accum_vert_cuts, min_total_penalty
