import heapq

# Helper function to initialize a graph with given vertices and edges
class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, u, v, weight):
        if u not in self.edges:
            self.edges[u] = []
        if v not in self.edges:
            self.edges[v] = []
        self.edges[u].append((v, weight))
        self.edges[v].append((u, weight))  # Assuming undirected edges

# Algorithm 1: Finding Pivots
def find_pivots(B, S, graph, k):
    W = set(S)  # Initially, W = S
    W0 = set(S)

    # Relaxation for k steps
    for i in range(1, k + 1):
        Wi = set()
        for u in W:
            for v, weight in graph.edges.get(u, []):
                if d_hat[u] + weight <= d_hat[v]:
                    d_hat[v] = d_hat[u] + weight
                if d_hat[u] + weight < B:
                    Wi.add(v)
        W.update(Wi)

        if len(W) > k * len(S):
            P = S  # Early termination
            return P, W

    # Creating the directed forest under Assumption 2.1
    F = {(u, v) for u in W for v, weight in graph.edges.get(u, []) if d_hat[v] == d_hat[u] + weight}
    P = {u for u in S if any(len(F_tree) >= k for F_tree in F)}

    return P, W

# Algorithm 2: Base Case of BMSSP
def base_case(B, S, graph, k):
    U0 = set(S)
    H = []  # Binary heap (priority queue)

    # Initialize with the first vertex of S
    x = next(iter(S))
    heapq.heappush(H, (d_hat[x], x))

    while H and len(U0) < k + 1:
        d, u = heapq.heappop(H)
        U0.add(u)

        for v, weight in graph.edges.get(u, []):
            if d_hat[u] + weight <= d_hat[v] and d_hat[u] + weight < B:
                d_hat[v] = d_hat[u] + weight
                if v not in [node for _, node in H]:
                    heapq.heappush(H, (d_hat[v], v))
                else:
                    heapq.heappush(H, (d_hat[v], v))

    if len(U0) <= k:
        return B, U0
    else:
        B_prime = max(d_hat[v] for v in U0)
        U = {v for v in U0 if d_hat[v] < B_prime}
        return B_prime, U


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.entry_finder = {}  # Maps each item to its entry in the queue
        self.REMOVED = '<removed-task>'  # Placeholder for a removed task
        self.counter = 0  # Unique sequence count

    def insert(self, item):
        'Add a new item or update the priority of an existing item'
        if item in self.entry_finder:
            self.remove(item)
        count = self.counter
        self.counter += 1
        entry = [item[0], count, item[1]]  # Priority, count, item
        self.entry_finder[item[1]] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, item):
        'Mark an existing item as REMOVED. Raise KeyError if not present.'
        entry = self.entry_finder.pop(item[1])
        entry[-1] = self.REMOVED

    def pull(self):
        'Remove and return the lowest priority task.'
        while self.queue:
            priority, count, item = heapq.heappop(self.queue)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return priority, item
        raise KeyError('pop from an empty priority queue')

    def __contains__(self, item):
        return item in self.entry_finder and self.entry_finder[item][-1] is not self.REMOVED

    def batch_prepend(self, tasks):
        'Insert multiple tasks into the priority queue'
        for task in tasks:
            self.insert(task)
# Algorithm 3: Bounded Multi-Source Shortest Path (BMSSP)
def bms_sp(l, B, S, graph, k):
    if l == 0:
        return base_case(B, S, graph, k)

    P, W = find_pivots(B, S, graph, k)

    D = PriorityQueue()  # Initialize D with the proper values
    for x in P:
        D.insert((d_hat[x], x))

    i = 0
    B0 = min(d_hat[x] for x in P) if P else B

    U = set()
    while len(U) < k * 2 * l and D:
        i += 1
        B_i, S_i = D.pull()
        B_prime, U_i = bms_sp(l - 1, B_i, S_i, graph, k)
        U.update(U_i)

        K = set()
        for u in U_i:
            for v, weight in graph.edges.get(u, []):
                if d_hat[u] + weight <= d_hat[v]:
                    d_hat[v] = d_hat[u] + weight
                if B_i <= d_hat[v] < B:
                    D.insert((d_hat[v], v))
                elif B_prime <= d_hat[v] < B_i:
                    K.add((v, d_hat[u] + weight))

        D.batch_prepend(K)

    return min(B_prime, B), U


# Usage of the functions
# Initialize graph
graph = Graph()
graph.add_edge(1, 2, 10)
graph.add_edge(2, 3, 20)
graph.add_edge(1, 3, 15)

# Define other parameters (you will need to define d_hat for each vertex)
d_hat = {1: 0, 2: float('inf'), 3: float('inf')}
S = {1, 2, 3}
B = 25
k = 2
l = 1

# Call BMSSP
result = bms_sp(l, B, S, graph, k)
print(result)
