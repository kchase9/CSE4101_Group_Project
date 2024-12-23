from queue import PriorityQueue
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

"""
A* Search Algorithm (adopted from https://www.geeksforgeeks.org/a-search-algorithm/)
Artificial Intelligence (ChatGPT 4o) was used to inform us of potential libraries and data objects to import, how to use pygame, and how to create multiple paths in a maze

1.  Initialize the open list
2.  Initialize the closed list
    put the starting node on the open 
    list (you can leave its f at zero)
3.  while the open list is not empty
    a) find the node with the least f on 
       the open list, call it "q"
    b) pop q off the open list
  
    c) generate q's 8 successors and set their 
       parents to q
   
    d) for each successor
        i) if successor is the goal, stop search
        
        ii) else, compute both g and h for successor
          successor.g = q.g + distance between 
                              successor and q
          successor.h = distance from goal to 
          successor (This can be done using many 
          ways, we will discuss three heuristics- 
          Manhattan, Diagonal and Euclidean 
          Heuristics)
          
          successor.f = successor.g + successor.h
        iii) if a node with the same position as 
            successor is in the OPEN list which has a 
           lower f than successor, skip this successor
        iV) if a node with the same position as 
            successor  is in the CLOSED list which has
            a lower f than successor, skip this successor
            otherwise, add  the node to the open list
     end (for loop)
  
    e) push q on the closed list
    end (while loop)
"""

# Create a maze
class Maze:
    def __init__(self, size: int = 12):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int) #1s are flames
        self.start = None
        self.goal = None
        
    def generate_maze(self):
        """Create a grid with random obstacles"""
        self.grid.fill(1)
        
        def paths(x, y):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)] # we move 2 spaces to avoid instances where 4 squares grouped with no walls between them
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                
                if (0 <= nx < self.size and 
                    0 <= ny < self.size and 
                    self.grid[nx, ny] == 1):

                        self.grid[x+dx//2, y+dy//2] = 0
                        self.grid[nx, ny] = 0
                        paths(nx, ny)

        startx = random.randrange(0, self.size, 2)
        starty = random.randrange(0, self.size, 2)
        self.grid[startx, starty] = 0
        paths(startx, starty)
        
        startx = random.randrange(0, self.size, 2)
        starty = random.randrange(0, self.size, 2)
        self.grid[startx, starty] = 0  # Mark starting cell as path
        paths(startx, starty)  # Generate the maze using backtracking

        # After maze generation, remove extra walls to create multiple paths
        for _ in range(self.size * 2):  # Randomly break more walls
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            if self.grid[x, y] == 1:  # If it's a wall, remove it
                self.grid[x, y] = 0
                
        # remove walls to make more paths
        # for i in range(self.size):
        #     x = random.randint(1, self.size - 2) #remove this, but claude intiially had 2
        #     y = random.randint(1, self.size - 2)
        
        
    def random_start(self):
        free_spaces = list()
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:
                    free_spaces.append((i, j))
        #maybe allow only edge pieces
        
        self.start = random.choice(free_spaces)
        
        free_spaces.remove(self.start) if self.start in free_spaces else None
        self.goal = random.choice(free_spaces)


class Agent:
    def __init__(self, maze):
        self.maze = maze
        
    def heuristic(self, pos):
        """ Calcualte the manhattan distance (our agent can only move up, dow, left or right)"""
        return abs(pos[0] - self.maze.goal[0]) + abs(pos[1] - self.maze.goal[1]) 
    
    def neighbours(self, pos):
        neighbours = list()
        
        for dx, dy in [(0, 1), (0, -1), (1,0), (-1, 0)]:
            nx, ny = pos[0]+dx, pos[1]+dy
            if (0 <= nx < self.maze.size and 0<= ny < self.maze.size and self.maze.grid[nx,ny]==0):
                neighbours.append((nx, ny))
        return neighbours
    
    def a_star(self):
        start = self.maze.start
        goal = self.maze.goal
        
        frontier = PriorityQueue()
        frontier.put((0, start))
        path_track = {start: None}
        cost_so_far = {start: 0}
        
        while not frontier.empty():
            current = frontier.get()[1]
            if current == goal:
                break
            
            for next in self.neighbours(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost<cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next)
                    frontier.put((priority, next))
                    path_track[next] = current
        
        path = []
        current = goal
        while current:
            path.append(current)
            current = path_track[current]
        path.reverse()    
        return path
    
class MazeScreen():
    def __init__(self, cell_size: int=30):
        self.cell_size = cell_size
        pygame.init()
        
    
    def visualize(self, maze: Maze, path):
        window_size = maze.size * self.cell_size
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Maze Environment")
        
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        
        screen.fill(WHITE)
        
        # Draw maze
        for i in range(maze.size):
            for j in range(maze.size):
                if maze.grid[i, j] == 1:
                    pygame.draw.rect(screen, BLACK, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
        
        # Draw path
        if path:
            for i in range(len(path) - 1):
                start_pos = (path[i][1] * self.cell_size + self.cell_size // 2, path[i][0] * self.cell_size + self.cell_size // 2)
                end_pos = (path[i + 1][1] * self.cell_size + self.cell_size // 2, path[i + 1][0] * self.cell_size + self.cell_size // 2)
                pygame.draw.line(screen, BLUE, start_pos, end_pos, 3)
        
        # Draw start and goal
        if maze.start:
            pygame.draw.circle(screen, GREEN, (maze.start[1] * self.cell_size + self.cell_size // 2, maze.start[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3)
        if maze.goal:
            pygame.draw.circle(screen, RED, (maze.goal[1] * self.cell_size + self.cell_size // 2, maze.goal[0] * self.cell_size + self.cell_size // 2), self.cell_size // 3)
        
        pygame.display.flip()
        
    def close(self):
        pygame.quit()
        
def main():
    maze = Maze(size = 12)
    
    for i in range(3):
        maze.generate_maze()
        maze.random_start()
        
        agent = Agent(maze)
        path = agent.a_star()
        
        # print the solutions found
        visualizer = MazeScreen()
        visualizer.visualize(maze, path)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        running = False
        
        visualizer.close()
        
if __name__ == "__main__":
    main()