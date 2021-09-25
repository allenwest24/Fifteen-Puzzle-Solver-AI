# Add these tests to the end on the main.py file to see the file work.
zero_moves = """1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 -"""

one_move = """1 2 3 4
5 6 7 8
9 10 11 12
13 14 - 15"""

six_moves = """1 2 3 4
5 10 6 8
- 9 7 12
13 14 11 15"""

sixteen_moves = """10 2 4 8
1 5 3 -
9 7 6 12
13 14 11 15"""

forty_moves = """4 3 - 11
2 1 6 8
13 9 7 15
10 14 12 5"""

solve_and_print(zero_moves, False)
solve_and_print(one_move, False)
solve_and_print(six_moves, False)

%time solve_and_print(sixteen_moves, False)
%time solve_and_print(forty_moves, False)
