"""Use A* to solve fifteen puzzle instances.

The "main" of this code is solve_and_print, at the end.  We'll try two different
heuristics, counting tiles out of place and summing Manhattan distance from
the destination over all tiles (the better heuristic)."""

import sys
import copy
import numpy as np
from queue import PriorityQueue
import math

PUZZLE_WIDTH = 4
BLANK = 0  # Integer comparison tends to be faster than string comparison

def read_puzzle_string(puzzle_string):
    """Read a NumberPuzzle from string representation; space-delimited, blank is "-".

    Args:
      puzzle_string (string):  string representation of the puzzle

    Returns:
      A NumberPuzzle
    """
    new_puzzle = NumberPuzzle()
    row = 0
    for line in puzzle_string.splitlines():
        tokens = line.split()
        for i in range(PUZZLE_WIDTH):
            if tokens[i] == '-':
                new_puzzle.tiles[row][i] = BLANK
                new_puzzle.blank_r = row
                new_puzzle.blank_c = i
            else:
                try:
                    new_puzzle.tiles[row][i] = int(tokens[i])
                except ValueError:
                    sys.exit("Found unexpected non-integer for tile value")
        row += 1
    return new_puzzle

class NumberPuzzle(object):
    """ Class containing the state of the puzzle, as well as A* bookkeeping info.

    Attributes:
        tiles (numpy array): 2D array of ints for tiles.
        blank_r (int):  Row of the blank, for easy identification of neighbors
        blank_c (int):  Column of blank, same reason
        parent (NumberPuzzle):  Reference to previous puzzle, for backtracking later
        dist_from_start (int):  Steps taken from start of puzzle to here
        key (int or float):  Key for priority queue to determine which puzzle is next
    """

    def __init__(self):
        """ Just return zeros for everything and fill in the tile array later"""
        self.tiles = np.zeros((PUZZLE_WIDTH, PUZZLE_WIDTH))
        self.blank_r = 0
        self.blank_c = 0
        # This next field is for our convenience when generating a solution
        # -- remember which puzzle was the move before
        self.parent = None
        self.dist_from_start = 0
        self.key = 0

    def __str__(self):
        """This is the Python equivalent of Java's toString()."""
        out = ""
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if j > 0:
                    out += " "
                if self.tiles[i][j] == BLANK:
                    out += "-"
                else:
                    out += str(int(self.tiles[i][j]))
            out += "\n"
        return out

    def copy(self):
        """Copy the puzzle and update the parent field.
        
        In A* search, we generally want to copy instead of destructively alter,
        since we're not backtracking so much as jumping around the search tree.
        Also, if A and B are numpy arrays, "A = B" only passes a reference to B.
        We'll also use this to tell the child we're its parent."""
        child = NumberPuzzle()
        child.tiles = np.copy(self.tiles)
        child.blank_r = self.blank_r
        child.blank_c = self.blank_c
        child.dist_from_start = self.dist_from_start
        child.parent = self
        return child

    def __eq__(self, other):
        """Governs behavior of ==.
        
        Overrides == for this object so that we can compare by tile arrangement
        instead of reference.  This is going to be pretty common, so we'll skip
        a type check on "other" for a modest speed increase"""
        return np.array_equal(self.tiles, other.tiles)

    def __hash__(self):
        """Generate a code for hash-based data structures.
        
        Hash function necessary for inclusion in a set -- unique "name"
        for this object -- we'll just hash the bytes of the 2D array"""
        return hash(bytes(self.tiles))

    def __lt__(self, obj):
        """Governs behavior of <, and more importantly, the priority queue.
        
        Override less-than so that we can put these in a priority queue
        with no problem.  We don't want to recompute the heuristic here,
        though -- that would be too slow to do it every time we need to
        reorganize the priority queue"""
        return self.key < obj.key

    def total_h(self, better_h):
        """A* cost:  admissible heuristic plus cost-so-far.

        Args:
            better_h (boolean):  True for Manhattan distance, false for counting tiles.
          
        Returns:
            A number representing the heuristic value (int or float)
        """
        return self.dist_from_start + self.heuristic(better_h)

    def move(self, tile_row, tile_column):
        """Move from the row, column coordinates given into the blank.

        Also very common, so we will also skip checks for legality to improve speed.

        Args:
            tile_row (int):  Row of the tile to move.
            tile_column (int):  Column of the tile to move.
        """

        self.tiles[self.blank_r][self.blank_c] = self.tiles[tile_row][tile_column]
        self.tiles[tile_row][tile_column] = BLANK
        self.blank_r = tile_row
        self.blank_c = tile_column
        self.dist_from_start += 1

    def legal_moves(self):
        """Return a list of NumberPuzzle states that could result from one move.

        Return a list of NumberPuzzle states that could result from one move
        on the present board.  Use this to keep the order in which
        moves are evaluated the same as our solution, thus matching the
        HackerRank solution as well.  (Also notice we're still in the
        methods of NumberPuzzle, hence the lack of arguments.)

        Returns:
            List of NumberPuzzles.
        """
        legal = []
        if self.blank_r > 0:
            down_result = self.copy()
            down_result.move(self.blank_r-1, self.blank_c)
            legal.append(down_result)
        if self.blank_c > 0:
            right_result = self.copy()
            right_result.move(self.blank_r, self.blank_c-1)
            legal.append(right_result)
        if self.blank_r < PUZZLE_WIDTH - 1:
            up_result = self.copy()
            up_result.move(self.blank_r+1, self.blank_c)
            legal.append(up_result)
        if self.blank_c < PUZZLE_WIDTH - 1:
            left_result = self.copy()
            left_result.move(self.blank_r, self.blank_c+1)
            legal.append(left_result)
        return legal

    def solve(self, better_h):
        """Return a list of puzzle states from this state to solved.

        Args:
            better_h (boolean):  True if Manhattan heuristic, false if tile counting

        Returns:
            path (list of NumberPuzzle or None) - path from start state to finish state
            explored - total number of nodes pulled from the priority queue
        """
        # Initialize open and closed list then add starting node on the open list.
        closed_list = set()
        open_list = PriorityQueue()
        open_list.put(self)

        # Total number of nodes pulled from the priority q.
        explored = 0

        # While open list is not empty.
        while open_list._qsize() != 0:
          explored += 1

          # Pull the current state and see it if requires more work.
          state = open_list._get()
          # Yes.
          if state.solved():
            goal_state = [state]
            parent = state.parent
            while parent is not None:
              goal_state.insert(0, parent)
              parent = parent.parent
            return goal_state, explored
          # No.
          # If a node with the same position as the successor is in the closed
          # list, skip the successor.
          q = state.__hash__()
          if q in closed_list:
            continue

          # Otherwise, add node to the open list.
          # Push q onto the closed list.
          for node in state.legal_moves():
            node.key = node.heuristic(better_h) + node.dist_from_start
            open_list.put(node)
            node.parent = state
          closed_list.add(q)

        return None, explored

    def solved(self):
        """"Return True iff all tiles in order and blank in bottom right."""
        should_be = 1
        for i in range(PUZZLE_WIDTH):
            for j in range(PUZZLE_WIDTH):
                if self.tiles[i][j] != should_be:
                    return False
                should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return True

    def heuristic(self, better_h):
        """Wrapper for the two heuristic functions.

        Args:
            better_h (boolean):  True if Manhattan heuristic, false if tile counting

        Returns:
            Value of the cost-to-go heuristic (int or float)
        """
        if better_h:
            return self.manhattan_heuristic()
        return self.tile_mismatch_heuristic()

    def tile_mismatch_heuristic(self):
        """Returns count of tiles out of place.
        
        Can't count the blank or it's inadmissible."""
        mismatch_count = 0
        # Completed. This was largely based on the solved() that returns a boolean.
        # Just adapted to show all the reasons it is not solved. 
        should_be = 1
        for ii in range(PUZZLE_WIDTH):
            for jj in range(PUZZLE_WIDTH):
                # Ignores the 0 tile in this line.
                if self.tiles[ii][jj] != should_be and self.tiles[ii][jj] != 0:
                    mismatch_count += 1
                should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return mismatch_count

    def manhattan_heuristic(self):
        """Returns total Manhattan (city block) distance from destination over all tiles.

        Again, shouldn't count blank; it gets where it's going for free."""
        total_manhattan = 0
        should_be = 1
        x_offset = y_offset = 0
        for ii in range(PUZZLE_WIDTH):
            for jj in range(PUZZLE_WIDTH):
                # Ignores the 0 tile in this line.
                if self.tiles[ii][jj] != should_be and self.tiles[ii][jj] != 0:
                  y_offset = abs(math.ceil(self.tiles[ii][jj] // 4) - ii)
                  x_offset = abs((self.tiles[ii][jj] % 4) - jj)
                  total_manhattan += x_offset + y_offset
                should_be = (should_be + 1) % (PUZZLE_WIDTH ** 2)
        return total_manhattan

    def path_to_here(self):
        """Returns list of NumberPuzzles giving the move sequence to get here.
        
        Retraces steps to this node through the parent fields."""
        path = []
        current = self
        while not current is None:
            path.insert(0, current)  # push
            current = current.parent
        return path

def print_steps(path):
    """ Print every puzzle in the path.

    Args:
        path (list of NumberPuzzle): list of puzzle states from start to finish
    """
    if path is None:
        print("No path found")
    else:
        print("{} steps".format(len(path)-1))
        for state in path:
            print(state)


def solve_and_print(puzzle_string : str, better_h : bool) -> None:
  """ "Main" - prints series of moves necessary to solve puzzle.

  Args:
    puzzle_string (string):  The puzzle to solve.
    better_h (boolean):  True if Manhattan distance heuristic, false if tile count
  """
  my_puzzle = read_puzzle_string(puzzle_string)
  solution_steps, explored = my_puzzle.solve(better_h) 
  print("{} nodes explored".format(explored))
  print_steps(solution_steps)
