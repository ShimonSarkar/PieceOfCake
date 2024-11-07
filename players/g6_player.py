import os
import pickle
from typing import List

import numpy as np
import logging
import traceback
import constants
import math
import itertools

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, tolerance: int) -> None:
        """Initialise the player with the basic information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                precomp_dir (str): Directory path to store/load pre-computation
                tolerance (int): tolerance for the cake distribution
                cake_len (int): Length of the smaller side of the cake
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        self.cuts = set()
        self.cutList =[]
        self.requests = None

    # Move the knife to the side in the clockwise direction
    def move_knife_clockwise(self, cur_pos):
        if cur_pos[0] == 0 and cur_pos[1] != 0:
            val = 0.01
            while [cur_pos[0], cur_pos[1], val, 0]  in self.cuts or [val, 0, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [val, 0]
        elif cur_pos[1] == 0 and cur_pos[0] != self.cake_width:
            val = 0.01
            while [cur_pos[0], cur_pos[1], self.cake_width, val] in self.cuts or [self.cake_width, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [self.cake_width, val]
        elif cur_pos[0] == self.cake_width and cur_pos[1] != self.cake_len:
            val = self.cake_width - 0.01
            while [cur_pos[0], cur_pos[1], val, self.cake_len] in self.cuts or [val, self.cake_len, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [val, self.cake_len]
        elif cur_pos[1] == self.cake_len and cur_pos[0] != 0:
            val = self.cake_len - 0.01
            while [cur_pos[0], cur_pos[1], 0, val] in self.cuts or [0, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [0, val]

    # Move the knife to the side in the anticlockwise direction
    def move_knife_anticlockwise(self, cur_pos):
        if cur_pos[0] == 0 and cur_pos[1] != self.cake_len:
            val = 0.01
            while [cur_pos[0], cur_pos[1], val, self.cake_len] in self.cuts or [val, self.cake_len, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [val, self.cake_len]
        elif cur_pos[1] == 0 and cur_pos[0] != 0:
            val = 0.01
            while [cur_pos[0], cur_pos[1], 0, val] in self.cuts or [0, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [0, val]
        elif cur_pos[0] == self.cake_width and cur_pos[1] != 0:
            val = self.cake_width - 0.01
            while [cur_pos[0], cur_pos[1], val, 0] in self.cuts or [val, 0, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [val, 0]
        elif cur_pos[1] == self.cake_len and cur_pos[0] != self.cake_width:
            val = self.cake_len - 0.01
            while [cur_pos[0], cur_pos[1], self.cake_width, val] in self.cuts or [self.cake_width, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [self.cake_width, val]
    
    def move_straight(self,cur_pos, dir='R'):
        switch ={
            'L': (0,cur_pos[1]),
            'R': (self.cake_width, cur_pos[1]),
            'U': (cur_pos[0], 0),
            'D': (cur_pos[0], self.cake_len)
        }
        return switch.get(dir)
    
    def move_angle(self, cur_pos, deg):
        deg = deg % 360
        theta_radians = np.radians(deg)

        dx = np.cos(theta_radians)
        dy = np.sin(theta_radians)

        intersection = None
        intersection = self.check_intersections(dx, dy, cur_pos)

        if intersection is None:
            # Take 180-angle i.e. line other side
            dx = np.cos(theta_radians + np.pi)
            dy = np.sin(theta_radians + np.pi)
            intersection = self.check_intersections(dx, dy, cur_pos)

        return intersection

    def check_intersections(self, dx, dy, cur_pos):
        L, W = self.cake_width, self.cake_len
        x0, y0 = cur_pos
        intersection = None

        # left side (x = 0)
        if dx != 0:
            tl = -x0 / dx
            if tl >= 0:
                y = round(y0 + tl * dy,2)
                if 0 <= y <= W and (0, y) != cur_pos:
                    intersection = (0, y)

        #  right side (x = L)
        if dx != 0:
            tr = (L - x0) / dx
            if tr >= 0:
                y = round(y0 + tr * dy,2)
                if 0 <= y <= W and (L, y) != cur_pos:
                    if intersection is None or tr < tl:
                        intersection = (L, y)

        # down side (y = 0)
        if dy != 0:
            td = -y0 / dy
            if td >= 0:
                x = round(x0 + td * dx,2)
                if 0 <= x <= L and (x, 0) != cur_pos:
                    if intersection is None or td < min(tl, tr):
                        intersection = (x, 0)

        #  up side (y = W)
        if dy != 0:
            tu = (W - y0) / dy
            if tu >= 0:
                x = round(x0 + tu * dx,2)
                if 0 <= x <= L and (x, W) != cur_pos:
                    if intersection is None or tu < min(tl, tr):
                        intersection = (x, W)

        return intersection

    def make_cuts(self):
        print(self.cake_len, self.cake_width)
        
        # TODO: If less than area do easy zigzag
        if self.cake_len*self.cake_width < 0:
            pass

        # TODO: Find number of vertical stacks
        num_ver_stacks = 4
        print(num_ver_stacks)

        #TODO: Find number of horizontal stacks based on vertical stacks
        num_stacks = 6

        areas = sorted(self.requests)

        #TODO: Handle extras
        if len(areas)%num_stacks != 0:
            pass

        print(areas)
        groups = {i: [] for i in range(num_stacks)}
        
        i=0
        j=len(self.requests)-1
        k=0

        while i<=j:
            # Small elements round robin
            for _ in range(min(num_stacks,j-i+1)):
                groups[k].append(areas[i])
                i+=1
                k=(k+1)%num_stacks
            
            # Large elements round robin
            for _ in range(min(num_stacks, j-i+1)):
                groups[k].append(areas[j])
                j-=1
                k=(k+1)%num_stacks
        
        # At this point i must be > j if we properly dealt with extras and all groups should have equal areas
        groups = {k: sorted(v) for k, v in groups.items()}
        print(groups)

        widths = [round(sum(groups[group])/self.cake_len,2) for group in groups]
        cum_widths = [round(s, 2) for s in itertools.accumulate(widths)]
        print(cum_widths)

        for i,w in enumerate(cum_widths):
            # Place knife
            if i==0:
                self.cutList.append([w,0])
                continue
            cur = self.cutList[-1]
            breadcrumb_goto =  0 if w<self.cake_width//2 else self.cake_width

            #Up to down
            if i%2 != 0:
                self.cutList.append(self.move_straight(cur, 'D'))
                self.cutList.append([breadcrumb_goto,round(self.cake_len-0.02,2)])
                self.cutList.append([w,self.cake_len])
            
            #Down to up
            else:
                self.cutList.append(self.move_straight(cur, 'U'))
                self.cutList.append([breadcrumb_goto,0.02])
                self.cutList.append([min(w, self.cake_width),0])

        if self.cutList[-1][0]!= self.cake_width:
            if self.cutList[-1][1] == 0:
                self.cutList.append(self.move_straight(self.cutList[-1], 'D'))
                self.cutList.append([breadcrumb_goto,round(self.cake_len-0.02,2)])
                self.cutList.append([round(self.cake_width-0.02,2),self.cake_len])
            else:
                self.cutList.append(self.move_straight(self.cutList[-1], 'U'))
                self.cutList.append([breadcrumb_goto,0.02])
                self.cutList.append([round(self.cake_width-0.02,2),0])
        
        #Horizontal cuts!
        l1 = [round(small/widths[i],2) for i,small in enumerate(groups[0][:-1])]
        l2 = [round(big/widths[i],2) for i,big in enumerate(groups[num_stacks-1][:-1])]
        cum_l1 = [round(s, 2) for s in itertools.accumulate(l1)]
        cum_l2 = [round(s, 2) for s in itertools.accumulate(l2)]
        print(cum_l1, cum_l2, l1, l2)

        for i,(l1,l2) in enumerate(zip(cum_l1,cum_l2)):                    
            #Right to left from l2 to l1
            if i%2==0:
               self.cutList.append([self.cake_width, l2])
               self.cutList.append([0, l1]) 
            # Left to right from l1 to l2
            else:
                self.cutList.append([0,l1])
                self.cutList.append([self.cake_width,l2])

            if i < len(cum_l1)-1:
                cur = self.cutList[-1]
                print(cur)
                breadcrumb_goto =  0 if cur[1]<self.cake_len//2 else self.cake_len
                if cur[0] == 0:
                    self.cutList.append([0.02, breadcrumb_goto])
                else:
                    self.cutList.append([round(self.cake_width-0.02,2), breadcrumb_goto])
        
        

    def move(self, current_percept) -> (int, List[int]):
        """Function which retrieves the current state of the cake

            Args:
                current_percept(PieceOfCakeState): contains current state information
            Returns:
                (int, List[int]): This function returns the next move of the user:
                The integer return value should be one of the following:
                    constants.INIT - If wants to initialize the knife position
                    constants.CUT - If wants to cut the cake
                    constants.ASSIGN - If wants to assign the pieces
        """
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width
        self.requests = requests

        try:
            if turn_number == 1:
                self.make_cuts()
                init = self.cutList.pop(0)
                return constants.INIT, [init[0],init[1]]

            while self.cutList:
                next_move = self.cutList.pop(0)
                return constants.CUT, [next_move[0], next_move[1]]

            if len(polygons) < len(requests):
                if cur_pos[0] == 0:
                    return constants.CUT, [self.cake_width, round((cur_pos[1] + 5)%self.cake_len, 2)]
                else:
                    return constants.CUT, [0, round((cur_pos[1] + 5)%self.cake_len, 2)]

            # Assign the pieces
            areas =[i.area for i in polygons]
            assignment = sorted(range(len(areas)), key=lambda x: areas[x], reverse=True)
            print(assignment[:len(requests)])
            return constants.ASSIGN, assignment[:len(requests)][::-1]
        
        except Exception as e:
            print(e)
            traceback.print_exc()

