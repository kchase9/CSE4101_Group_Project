from queue import PriorityQueue
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

"""
A* Search Algorithm (https://www.geeksforgeeks.org/a-search-algorithm/)
Artificial Intelligence (ChatGPT 4o) was used to inform us of potential libraries and data objects to import, how to use pygame, and how to create multiple paths in a python maze

"""

# Create a maze
class Maze:
    def __init__(self, size: int = 12):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int) #1s are flames
        self.start1 = None
        self.start2 = None
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
        paths(startx, starty) 

        for p in range(self.size * 2):  
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            if self.grid[x, y] == 1: 
                self.grid[x, y] = 0
                
        
    def random_start(self): # change to allow 2 start positions (agents)
        # free_spaces = list()
        # for i in range(self.size):
        #     for j in range(self.size):
        #         if self.grid[i, j] == 0:
        #             free_spaces.append((i, j))
     
        
        # self.start = random.choice(free_spaces)
        
        # free_spaces.remove(self.start) if self.start in free_spaces else None
        # self.goal = random.choice(free_spaces)
        free_spaces = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 0]
    
        self.start1 = random.choice(free_spaces)
        free_spaces.remove(self.start1)
        self.start2 = random.choice(free_spaces)
        free_spaces.remove(self.start2)
        self.goal = random.choice(free_spaces)


class Multi_Agent:
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
    
    def a_star_besties(self, other_agent, visualize_callback):
        start1 = self.maze.start1
        start2 = self.maze.start2
        goal = self.maze.goal

        # Separate frontiers
        frontier1 = PriorityQueue()
        frontier2 = PriorityQueue()
        frontier1.put((0, start1))
        frontier2.put((0, start2))
        
        # Separate path and cost tracking
        path_track1 = {start1: None}
        path_track2 = {start2: None}
        cost_so_far1 = {start1: 0}
        cost_so_far2 = {start2: 0}
        
        # Shared explored set for cooperation
        shared_explored = set()
        
        path1 = [start1]
        path2 = [start2]

        # While either agent still has nodes to explore
        while not frontier1.empty() or not frontier2.empty():
            # Agent 1
            if not frontier1.empty():
                current1 = frontier1.get()[1]
                path1.append(current1)
                shared_explored.add(current1)

                if current1 == goal:
                    return path1, path2

                for next1 in self.neighbours(current1):
                    if next1 not in shared_explored:  # Use shared knowledge
                        new_cost = cost_so_far1[current1] + 1
                        if next1 not in cost_so_far1 or new_cost < cost_so_far1[next1]:
                            cost_so_far1[next1] = new_cost
                            priority = new_cost + self.heuristic(next1)
                            frontier1.put((priority, next1))
                            path_track1[next1] = current1

            # Agent 2
            if not frontier2.empty():
                current2 = frontier2.get()[1]
                path2.append(current2)
                shared_explored.add(current2)

                if current2 == goal:
                    return path1, path2

                for next2 in self.neighbours(current2):
                    if next2 not in shared_explored:  # Use shared knowledge
                        new_cost = cost_so_far2[current2] + 1
                        if next2 not in cost_so_far2 or new_cost < cost_so_far2[next2]:
                            cost_so_far2[next2] = new_cost
                            priority = new_cost + self.heuristic(next2)
                            frontier2.put((priority, next2))
                            path_track2[next2] = current2

            # Update screen
            visualize_callback(path1, path2)

        return path1, path2

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class MazeScreen:
    def __init__(self, cell_size: int = 30):
        self.cell_size = cell_size
        pygame.init()

    def visualize_progress(self, maze, path1, path2):
        window_size = maze.size * self.cell_size
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Cooperative Maze Environment")

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        BRIGHTBLUE = (0, 0, 255)
        DARKERBLUE = (0, 0, 127)
        PATH_COLOR1 = (0, 255, 0)  # Green for Agent 1
        PATH_COLOR2 = (255, 165, 0)  # Orange for Agent 2

        screen.fill(WHITE)

        # Draw maze
        for i in range(maze.size):
            for j in range(maze.size):
                if maze.grid[i, j] == 1:
                    pygame.draw.rect(screen, BLACK, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        if path1:
            for pos in path1:
                pygame.draw.rect(screen, PATH_COLOR1, (pos[1] * self.cell_size + self.cell_size // 4,pos[0] * self.cell_size + self.cell_size // 4,self.cell_size // 2, self.cell_size // 2))
        if path2:
            for pos in path2:
                pygame.draw.rect(screen, PATH_COLOR2, (pos[1] * self.cell_size + self.cell_size // 4,pos[0] * self.cell_size + self.cell_size // 4,self.cell_size // 2, self.cell_size // 2))

        if maze.start1:
            pygame.draw.circle(screen, BRIGHTBLUE, (maze.start1[1] * self.cell_size + self.cell_size // 2,maze.start1[0] * self.cell_size + self.cell_size // 2),self.cell_size // 3)
        if maze.start2:
            pygame.draw.circle(screen, DARKERBLUE, (maze.start2[1] * self.cell_size + self.cell_size // 2,maze.start2[0] * self.cell_size + self.cell_size // 2),self.cell_size // 3)
        if maze.goal:
            pygame.draw.circle(screen, RED, (maze.goal[1] * self.cell_size + self.cell_size // 2,maze.goal[0] * self.cell_size + self.cell_size // 2),self.cell_size // 3)

        pygame.display.flip()
        pygame.time.wait(200)  # Delay

    def close(self):
        pygame.quit()


      
def main():
    maze = Maze(size=16)

    for i in range(3):  # 3 Scenarios
        maze.generate_maze()
        maze.random_start()
        start_time = time.time()
        agent1 = Multi_Agent(maze)
        agent2 = Multi_Agent(maze)

        visualizer = MazeScreen()

        def update_visualization(path1, path2):
            visualizer.visualize_progress(maze, path1, path2)

        path1, path2 = agent1.a_star_besties(agent2, update_visualization)  #call the maze visualize after every step
        end_time = time.time()

        duration = end_time - start_time
        print(f"Execution time for maze {i + 1}: {duration:.2f} seconds")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    running = False

        visualizer.close()

if __name__ == "__main__":
    main()
