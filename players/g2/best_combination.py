from typing import Callable
from shapely.geometry import Polygon
from tqdm import tqdm
import constants
from players.g2.assigns import (
    sorted_assign,
    hungarian_min_penalty,
    greedy_best_fit_assignment,
)
from players.g2.helpers import sneak, divide_polygon, can_cake_fit_in_plate
import random


def get_cuts_spread(requests: list[float]) -> tuple[int, int]:
    count = len(requests)

    # the absolute maximum number of cuts required to match the number of requests
    # in practice, this should never be reached
    max_cuts = count - 1

    # an average cut will create 3 new slices
    # * Actually, its dependent on the number of existing cuts,
    # * the more existing cuts on the cake, the more slices will be created
    # * from the next cut
    min_cuts = count // 3

    # ensure max_cuts > min_cuts
    max_cuts = max(max_cuts, min_cuts + 1)

    return min_cuts, max_cuts


def generate_cuts(
    cuts: int,
    cake_len: float,
    cake_width: float,
    jumps: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    # how many different points should be considered on each edge
    GAP = jumps - 1
    LEFT = [(0, round(cake_len * i / GAP, 2)) for i in range(jumps)]
    RIGHT = [(cake_width, round(cake_len * i / GAP, 2)) for i in range(jumps)]
    UP = [(round(cake_width * i / GAP, 2), 0) for i in range(jumps)]
    DOWN = [(round(cake_width * i / GAP, 2), cake_len) for i in range(jumps)]

    points = set(LEFT + RIGHT + UP + DOWN)

    selected: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    while len(selected) < cuts:
        from_point = random.sample(list(points), 1)[0]

        if from_point in LEFT:
            to_point = random.sample(list(points.difference(LEFT)), 1)[0]
        elif from_point in RIGHT:
            to_point = random.sample(list(points.difference(RIGHT)), 1)[0]
        elif from_point in UP:
            to_point = random.sample(list(points.difference(UP)), 1)[0]
        else:
            to_point = random.sample(list(points.difference(DOWN)), 1)[0]

        if (from_point, to_point) not in selected and (
            to_point,
            from_point,
        ) not in selected:
            selected.add((from_point, to_point))

    return list(selected)


def __calculate_penalty(
    assign_func: Callable[[list[Polygon], list[float]], list[int]],
    requests: list[float],
    polygons: list[Polygon],
    tolerance,
) -> float:
    penalty = 0
    assignments: list[int] = assign_func(polygons, requests, tolerance)

    for request_index, assignment in enumerate(assignments):
        # check if the cake piece fit on a plate of diameter 25 and calculate penaly accordingly
        if assignment == -1 or (not can_cake_fit_in_plate(polygons[assignment])):
            penalty += 100
        else:
            penalty_percentage = (
                100
                * abs(polygons[assignment].area - requests[request_index])
                / requests[request_index]
            )
            if penalty_percentage > tolerance:
                penalty += penalty_percentage
    return penalty


def cuts_to_polygons(cuts: list, cake_len: float, cake_width: float) -> list[Polygon]:
    polygons = [
        Polygon(
            [
                (0, 0),
                (0, cake_len),
                (cake_width, cake_len),
                (cake_width, 0),
            ]
        )
    ]

    for cut in cuts:
        from_point = cut[0]
        to_point = cut[1]
        new_polygons = []
        for polygon in polygons:
            new_polygons.extend(divide_polygon(polygon, from_point, to_point))

        polygons = new_polygons

    return polygons


def penalty(
    cuts: list[tuple[tuple[float, float], tuple[float, float]]],
    requests: list[float],
    cake_len,
    cake_width,
    tolerance,
):
    polygons = cuts_to_polygons(cuts, cake_len, cake_width)
    return __calculate_penalty(
        greedy_best_fit_assignment, requests, polygons, tolerance
    )


def best_combo(
    requests: list[float],
    cake_len: float,
    cake_width: float,
    tolerance: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    # 1. SEARCH
    # search for optimal no. of cuts
    print("\n1. SEARCH\nsearch optimal no. of cuts")
    min_cuts, max_cuts = get_cuts_spread(requests)

    best_cuts = []
    min_penalty = curr_penalty = float("inf")
    best_cut_no = min_cuts
    for cuts in range(min_cuts, max_cuts + 1):
        # we've probably crossed the optimal no. of cuts
        if best_cut_no + 5 < cuts:
            break
        print(f"{cuts} cuts: ", end="")

        curr_best_cuts = []
        best_contender = best_curr_penalty = float("inf")
        # try 100 combinations for each cut,
        # use best one
        for _ in range(200):
            cuts_contender = generate_cuts(cuts, cake_len, cake_width, 6)
            curr_penalty = penalty(
                cuts_contender, requests, cake_len, cake_width, tolerance
            )

            if not curr_best_cuts or curr_penalty < best_curr_penalty:
                best_contender = cuts_contender
                best_curr_penalty = curr_penalty

        print(f"{round(best_curr_penalty, 2)} penalty")
        if not best_cuts or best_curr_penalty < min_penalty:
            best_cuts = best_contender
            min_penalty = best_curr_penalty
            best_cut_no = cuts

    # 2. SPAM
    # found the optimal no. of cuts
    # spam combinations for that number
    print(f"\n 2. SPAM\nspamming optimal cut ({best_cut_no})")
    for _ in tqdm(range(2000)):
        cuts_contender = generate_cuts(cuts, cake_len, cake_width, 12)
        curr_penalty = penalty(
            cuts_contender, requests, cake_len, cake_width, tolerance
        )

        if curr_penalty < min_penalty:
            tqdm.write(f"found lower penalty ({round(curr_penalty, 2)})")
            best_cuts = cuts_contender
            min_penalty = curr_penalty

    # 3. SHAKE
    # shift line's around in the optimal set
    # of cuts for slightly lower penalties
    best_cuts = shake(best_cuts, requests, min_penalty, cake_len, cake_width, tolerance)

    return best_cuts


def create_offspring(cuts, c1, c2, cake_len, cake_width):
    offspring = []
    MUTATION_PROB = 0.4
    MUTATION_DIST = 0.1

    for idx in range(len(c1)):
        toss = random.random()
        if toss < 0.5:
            genome = [i.copy() for i in c1[idx]]
        else:
            genome = [i.copy() for i in c2[idx]]

        mut = random.random()
        if mut < MUTATION_PROB:
            l, r = cuts[idx]
            # mutate left point
            if mut < MUTATION_PROB / 2:
                # which axis in the point can we modify while
                # ensuring the point remains on the cake border
                if l[0] == 0 or l[0] == cake_width:  # l is on LEFT | RIGHT border
                    if (
                        mut < MUTATION_PROB / 4
                        and l[1] + genome[0][1] + MUTATION_DIST < cake_len
                    ):
                        genome[0][1] += MUTATION_DIST
                    elif 0 < l[1] + genome[0][1] - MUTATION_DIST:
                        genome[0][1] -= MUTATION_DIST
                elif l[1] == 0 or l[1] == cake_len:  # l is on TOP | BOTTOM border
                    if (
                        mut < MUTATION_PROB / 4
                        and l[0] + genome[0][0] + MUTATION_DIST < cake_width
                    ):
                        genome[0][0] += MUTATION_DIST
                    elif 0 < l[0] + genome[0][0] - MUTATION_DIST:
                        genome[0][0] -= MUTATION_DIST

            # mutate right point
            else:
                # which axis in the point can we modify while
                # ensuring the point remains on the cake border
                if r[0] == 0 or r[0] == cake_width:  # r is on LEFT | RIGHT border
                    if (
                        mut - MUTATION_PROB / 2 < MUTATION_PROB / 4
                        and r[1] + genome[1][1] + MUTATION_DIST < cake_len
                    ):
                        genome[1][1] += MUTATION_DIST
                    elif 0 < r[1] + genome[1][1] - MUTATION_DIST:
                        genome[1][1] -= MUTATION_DIST
                elif r[1] == 0 or r[1] == cake_len:  # r is on TOP | BOTTOM border
                    if (
                        mut - MUTATION_PROB / 2 < MUTATION_PROB / 4
                        and r[1] + genome[1][0] + MUTATION_DIST < cake_len
                    ):
                        genome[1][0] += MUTATION_DIST
                    elif 0 < r[0] + genome[1][0] - MUTATION_DIST:
                        genome[1][0] -= MUTATION_DIST

        genome = [[round(num, 2) for num in sublist] for sublist in genome]
        offspring.append(genome)

    return offspring


def combined_cuts(cuts, candidate):
    return [
        [
            [
                round(cuts[idx][0][i] + candidate[idx][0][i], 2)
                for i in range(len(cuts[idx]))
            ],
            [
                round(cuts[idx][1][i] + candidate[idx][1][i], 2)
                for i in range(len(cuts[idx]))
            ],
        ]
        for idx in range(len(cuts))
    ]


def sort_candidates(cuts, candidates, requests, cake_len, cake_width, tolerance):
    candidates.sort(
        key=lambda c: penalty(
            combined_cuts(cuts, c), requests, cake_len, cake_width, tolerance
        )
    )


def shake(
    cuts: list[tuple[tuple[float, float], tuple[float, float]]],
    requests,
    pen: float,
    cake_len: float,
    cake_width: float,
    tolerance,
):
    # number of candidates in tribe
    NUM_CANDIDATES = 20
    # number of candidates cut off after each epoc
    CUTOFF = 6
    # number of epocs we must pass without evolving until we terminate
    MAX_UNCHANGED_EPOCS = 30

    candidates = [[[[0.0, 0.0], [0.0, 0.0]] for _ in range(len(cuts))]] * 2

    # initialize population
    while len(candidates) < NUM_CANDIDATES:
        offspring = create_offspring(
            cuts, *random.sample(candidates, 2), cake_len, cake_width
        )
        candidates.append(offspring)

    sort_candidates(cuts, candidates, requests, cake_len, cake_width, tolerance)
    candidates = candidates[:-CUTOFF]  # cut worst candidates

    min_penalty = pen
    best_epoc = 0
    curr_epoc = 0
    while 0 < min_penalty and curr_epoc < best_epoc + MAX_UNCHANGED_EPOCS:
        while len(candidates) < NUM_CANDIDATES:
            offspring = create_offspring(
                cuts, *random.sample(candidates, 2), cake_len, cake_width
            )
            candidates.append(offspring)

        sort_candidates(cuts, candidates, requests, cake_len, cake_width, tolerance)
        candidates = candidates[:-CUTOFF]  # cut worst candidates
        print(
            f"epoc {curr_epoc}: {round(penalty(cuts, requests, cake_len, cake_width, tolerance), 2)} -> {round(penalty(combined_cuts(cuts, candidates[0]), requests, cake_len, cake_width, tolerance), 2)}"
        )

        curr_epoc += 1
        if (
            curr_penalty := penalty(
                combined_cuts(cuts, candidates[0]),
                requests,
                cake_len,
                cake_width,
                tolerance,
            )
        ) < min_penalty:
            min_penalty = curr_penalty
            best_epoc = curr_epoc

    return combined_cuts(cuts, candidates[0])


def cuts_to_moves(
    cuts: list[tuple[tuple[float, float], tuple[float, float]]],
    requests,
    cake_len,
    cake_width,
) -> list[tuple[int, list]]:
    moves = []
    last_point = None
    for cut in cuts:
        from_point = list(cut[0])
        to_point = list(cut[1])
        if not moves:
            moves.append((constants.INIT, from_point))
        else:
            sneak_points = sneak(last_point, from_point, cake_width, cake_len)
            for point in sneak_points:
                # moves.append((constants.CUT, [last_point, point]))
                moves.append((constants.CUT, point))
                last_point = point

        # moves.append((constants.CUT, [from_point, to_point]))
        moves.append((constants.CUT, to_point))
        last_point = to_point

    return moves
