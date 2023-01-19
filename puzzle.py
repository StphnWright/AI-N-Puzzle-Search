#!/usr/bin/env python
# coding: utf-8

# In[12]:


from __future__ import division
from __future__ import print_function

import sys
import math
import time
import resource
import heapq


# In[13]:


## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []
        self.num_tiles = n * n
        
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])
    
    def move_up(self):
        """
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        move_index = self.blank_index - self.n
        if (self.action == "Down") or (move_index < 0):
            return None
        else:
            return self.get_move_child(move_index, "Up")
    
    def move_down(self):
        """ 
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        move_index = self.blank_index + self.n
        if (self.action == "Up") or (move_index > (self.num_tiles - 1)):
            return None
        else:
            return self.get_move_child(move_index, "Down")
    
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        if (self.action == "Right") or (self.blank_index % self.n == 0): 
            return None
        else:
            move_index = self.blank_index - 1
            return self.get_move_child(move_index, "Left")
    
    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        if (self.action == "Left") or (self.blank_index % self.n == self.n - 1): 
            return None
        else:
            move_index = self.blank_index + 1
            return self.get_move_child(move_index, "Right")
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
            
        return self.children
    
    def get_move_child(self, move_index, action_value):
        config_new = list(self.config)
        config_new[self.blank_index] = self.config[move_index]
        config_new[move_index] = 0
        return PuzzleState(config_new, self.n, parent=self, action=action_value, cost=self.cost + 1)
    
    def __eq__(self, other):
        return (self.config == other.config)

    def __hash__(self):
        return hash(str(self.config))


# In[33]:


## Frontier class
class Frontier(object):
    """
        Base class for frontier objects, in this case a queue
        The items structure is a list by default
        A set is used internally for fast membership checking
    """
    def __init__(self):
        self.items = []
        self.items_set = set()
        
    def __contains__(self, state):
        return (state in self.items_set)
    
    def __len__(self):
        return len(self.items_set)
    
    def empty(self):
        return (len(self.items_set) == 0)
    
    def get(self):
        state = self.items.pop(0)
        self.items_set.remove(state)
        return state
    
    def put(self, state):
        self.items.append(state)
        self.items_set.add(state)   

class FrontierQueue(Frontier):
    """
        FIFO queue for the frontier
    """
    def __init__(self):
        super().__init__()

class FrontierStack(Frontier):
    """
        Stack (LIFO queue) for the frontier
    """
    def __init__(self):
        super().__init__()
    
    def put(self, state):
        self.items.insert(0, state) 
        self.items_set.add(state)
        
class FrontierHeap(Frontier):
    """
        Heap (priority queue) for the frontier
        Items on the heap are structured as a three element tuple (A, B, C) where:
        - A = cost of the state (from calculate_total_cost)
        - B = order in which the state was added to the heap
        - C = the state
        Higher priority is given to lower values of A, and as a tiebreaker lower
        values of B (that is, states added earlier to the heap get higher priority)
    """
    def __init__(self):
        self.items = []
        self.items_set = set()
        self.n = 0
    
    def get(self):
        item = heapq.heappop(self.items)
        self.items_set.remove(item[2])
        return item[2]
    
    def put(self, state, priority):
        heapq.heappush(self.items, (priority, self.n, state))
        self.items_set.add(state)
        self.n += 1
        
    def decrease_key(self, state, new_priority):        
        # Find the first (highest-priority) instance of this state in the heap
        for i in range(len(self.items)):
            if (self.items[i][2] == state):
                break
        
        # Add the new state if it has a higher priority than the one just found
        if (new_priority < self.items[i][0]):
            # Remove the state at position i from the set
            self.items_set.remove(self.items[i][2])
            
            # Delete the state at position i from the heap by replacing it with 
            # the last item in the heap and then popping the heap
            self.items[i] = self.items[-1]
            self.items.pop()
            
            # Apply sifting to restore the heap invariant
            if (i < len(self.items) - 1):
                heapq._siftup(self.items, i)
                heapq._siftdown(self.items, 0, i)
            
            # Now add the new state to the heap
            self.put(state, new_priority)


# In[41]:


# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(goal_state, nodes_expanded, max_search_depth, time_used, ram_used):
    search_depth = goal_state.depth
    cost_of_path = goal_state.cost
    
    state = goal_state
    path_to_goal_list = []
    
    while True:
        if (state.parent is None):
            break
        else:
            path_to_goal_list.append(state.action)
            state = state.parent
    
    # Generate path string
    path_to_goal = ''
    if (len(path_to_goal_list) > 0):
        for action in reversed(path_to_goal_list):
            path_to_goal += ", '" + action + "'"
        
        # Remove first comma and space
        path_to_goal = path_to_goal[2:]
    
    # Enclose in square brackets
    path_to_goal = "[" + path_to_goal + "]"
        
    # Write the output
    str_time_used = f'{time_used:.8f}'
    str_ram_used = f'{ram_used:.8f}'
    output_text = ['path_to_goal: ' + path_to_goal, 
                   'cost_of_path: ' + str(cost_of_path),
                   'nodes_expanded: ' + str(nodes_expanded),
                   'search_depth: ' + str(search_depth),
                   'max_search_depth: ' + str(max_search_depth),
                   'running_time: ' + str_time_used, 
                   'max_ram_usage: ' + str_ram_used]
        
    with open('output.txt', 'w') as f:
        for line in output_text:
            f.write(line + '\n')
    
    for line in output_text:
        if (len(line) > 500):
            print(line[0:500] + '...')
        else:
            print(line)

def bfs_search(initial_state):
    """BFS search"""
    return main_search(initial_state, search_method = "bfs")

def dfs_search(initial_state):
    """DFS search"""
    return main_search(initial_state, search_method = "dfs")

def A_star_search(initial_state):
    """A * search"""
    return main_search(initial_state, search_method = "ast")

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    cost = state.cost
    for k in range(state.num_tiles):
        if (k != state.blank_index):
            cost += calculate_manhattan_dist(k, state.config[k], state.n)
    return cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile
    :param idx: index of tile (i.e., where the tile is currently)
    :param value: value of the tile, which is also the goal index (i.e., where the tile should be)
    :param n: # of rows
    """
    yc = idx % n
    xc = (idx - yc) / n
    yt = value % n
    xt = (value - yt) / n
    return (abs(xc - xt) + abs(yc - yt))

def main_search(initial_state, search_method = "bfs"):
    ram_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()
    goal_config = list(range(initial_state.num_tiles))   
    
    is_dfs = (search_method == "dfs")
    is_ast = (search_method == "ast")
    
    if is_dfs:
        frontier = FrontierStack()
        frontier.put(initial_state)
    elif is_ast:
        frontier = FrontierHeap()
        frontier.put(initial_state, calculate_total_cost(initial_state))
    else:
        frontier = FrontierQueue()
        frontier.put(initial_state)
        
    explored = set()
    nodes_expanded = 0
    max_search_depth = 0
    
    while not frontier.empty():
        state = frontier.get()
        explored.add(state)
        
        if (state.config == goal_config):
            time_used = time.time() - start_time
            ram_used = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - ram_start) / (2**20)
            writeOutput(state, nodes_expanded, max_search_depth, time_used, ram_used)
            return True
        
        children = state.expand()
        nodes_expanded += 1
        
        if (len(children) != 0) and (state.depth + 1 > max_search_depth):
            max_search_depth = state.depth + 1
            if (max_search_depth % 100 == 0):
                print(str(max_search_depth) + " levels / " + str(nodes_expanded) + " expansions", end='\r')
        
        if is_ast:
            for child in children:
                if (child in frontier):
                    frontier.decrease_key(child, calculate_total_cost(child))
                elif (child not in explored):
                    frontier.put(child, calculate_total_cost(child))
        else:
            if is_dfs:
                children = reversed(children)
            
            for child in children:
                if (child not in explored) and (child not in frontier):
                    frontier.put(child)
        
    print("Failure")
    return False

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
    