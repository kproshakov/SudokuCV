
class SudokuSolver():
    def __init__(self, grid):
        self.grid = grid

    def is_possible(self, x, y, n):
        for i in range(9):
            if self.grid[i][x] == n and i != y:
                return False

        for i in range(9):
            if self.grid[y][i] == n and i != x:
                return False

        offset_x, offset_y = x//3, y//3

        for i in range(3):
            for j in range(3):
                if self.grid[offset_y*3 + i][offset_x*3 + j] == n and (offset_y*3 + i != y and offset_x*3 + j != x):
                    return False
        return True


    def is_valid(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y][x] != 0:
                    if not self.is_possible(x, y, self.grid[y][x]):
                        return False
        return True

    def solve(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y][x] == 0:
                    available = False
                    for n in range(9):
                        if self.is_possible(x, y, n+1):
                            self.grid[y][x] = n+1
                            if self.solve():
                                available = True
                                break
                            self.grid[y][x] = 0
                    if not available:
                        return False
        return True
    def get_grid(self):
        return self.grid


