import matplotlib.pyplot as plt
import numpy as np
import collections
import plotly.graph_objects as go
from osmnx import distance as distance
import networkx as nx
import osmnx as ox
from operator import itemgetter
import time

ox.config(log_console=True, use_cache=True)

G = ox.graph_from_address(
    '1276 Gilbreath Drive, Johnson City, TN, US', dist=4000, network_type='drive')

# Use this code to display a plot of the graph if desired. Note: You need to import matplotlib.pyplot as plt
fig, ax = ox.plot_graph(G, edge_linewidth=3,
                        node_size=0, show=False, close=False)
plt.show()

# Class provided by Dr. Brian Bennett
# CSCI-5260-940, lab 2, search_examples.py


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return self.queue.__iter__()

    def set_priority(self, item, priority):
        for node in self.queue:
            if node[0] == item:
                self.queue.remove(node)
                break
        self.put(item, priority)

    def put(self, item, priority):
        node = [item, priority]
        self.queue.append(node)
        self.queue.sort(key=itemgetter(1))

    def get(self):
        if len(self.queue) == 0:
            return None
        node = self.queue.pop(0)
        return node[0]

    def isEmpty(self):
        return len(self.queue) == 0


class Queue:
    def __init__(self):
        self.queue = collections.deque()

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return self.queue.__iter__()

    def put(self, item):
        self.queue.appendleft(item)

    def get(self):
        return self.queue.pop()

    def isEmpty(self):
        return len(self.queue) == 0


class Stack:
    def __init__(self):
        self.stack = collections.deque()

    def __len__(self):
        return len(self.stack)

    def __iter__(self):
        return self.stack.__iter__()

    def put(self, item):
        self.stack.append(item)

    def get(self):
        return self.stack.pop()

    def isEmpty(self):
        return len(self.stack) == 0


def node_list_to_path(gr, node_list):
    """
    SOURCE: Modified from Priyam, Apurv (2020). https://towardsdatascience.com/find-and-plot-your-optimal-path-using-plotly-and-networkx-in-python-17e75387b873

    Given a list of nodes, return a list of lines that together
    follow the path
    defined by the list of nodes.
    Parameters
    ----------
    gr : networkx multidigraph
    node_list : list
        the route as a list of nodes
    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start),
    (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))

    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        if gr.get_edge_data(u, v) is not None:
            data = min(gr.get_edge_data(u, v).values(),
                       key=lambda x: x['length'])

            # if it has a geometry attribute
            if 'geometry' in data:
                # add them to the list of lines to plot
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
            else:
                # if it doesn't have a geometry attribute,
                # then the edge is a straight line from node to node
                x1 = gr.nodes[u]['x']
                y1 = gr.nodes[u]['y']
                x2 = gr.nodes[v]['x']
                y2 = gr.nodes[v]['y']
                line = [(x1, y1), (x2, y2)]
                lines.append(line)

    return lines


def plot_path(lat, long, origin_point, destination_point):
    """
    SOURCE: Modified from Priyam, Apurv (2020). https://towardsdatascience.com/find-and-plot-your-optimal-path-using-plotly-and-networkx-in-python-17e75387b873

    Given a list of latitudes and longitudes, origin
    and destination point, plots a path on a map

    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    """
    origin = (origin_point[1]["y"], origin_point[1]["x"])
    destination = (destination_point[1]["y"], destination_point[1]["x"])
    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker={'size': 10},
        line=dict(width=4.5, color='blue')))
    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name="Source",
        mode="markers",
        lon=[origin[1]],
        lat=[origin[0]],
        marker={'size': 12, 'color': "red"}))

    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name="Destination",
        mode="markers",
        lon=[destination[1]],
        lat=[destination[0]],
        marker={'size': 12, 'color': 'green'}))

    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
                      mapbox_center_lat=30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'center': {'lat': lat_center,
                                     'lon': long_center},
                          'zoom': 13})
    fig.show()


def metrics(graph, path, explored):
    '''
    Accepts a graph and a list of node Id's that represent a path
    between two nodes.

    It walks through each node along the path to determine each
    node's id/lat/long/distance to next node.

    This method returns a tuple in the following format:
        (id[], lat[], long[], totalDist)
    Where the id[] is a list of ids along the path,
    lat[] is the latitude of each node along the path,
    long[] is the longitude of each node along the path,
    and totalDist is the path cost, aka total distance,
    traveled along the path.
    '''
    print("Calculating path metrics")
    lat = []
    long = []

    for nodeId in path:
        node = graph.nodes[nodeId]
        lat.append(node["y"])
        long.append(node["x"])

    totalDistance = 0
    for i in range(len(path)-1):
        localDistance = getDistance(graph, path[i], path[i+1])
        totalDistance += localDistance

    steps = len(explored)

    return (path, lat, long, totalDistance, steps)


def getDistance(graph, nodeAId, nodeBId):
    nodeA = graph.nodes[nodeAId]
    nodeB = graph.nodes[nodeBId]
    radiusOfEarth_miles = 3963.1906
    return ox.distance.great_circle_vec(nodeA["x"], nodeA["y"], nodeB["x"], nodeB["y"], radiusOfEarth_miles)


def backtrack(graph, destinationNodeId, exploredDictionary):
    '''
    Accepts the graph, the destination node's ID, and a dictionary of the 
    explored nodes in the format (nodeId, nodeId), where the key is a given
    explored node, and the value is the node which added it to the explored
    dictionary.

    It should work backwards to the origin from node, using explored, so you
    know exactly which path you took to locate the destination.

    You should return a list of osmids in the order of the path.
    It may also be helpful to track the latitudes and longitudes of each item on
    the path so you can easily move it to the plot function, along with the
    path cost (i.e., distance).
    '''
    print("Backtracking")
    # Start assembling the path back to the origin, starting with the destination
    path = [destinationNodeId]
    previous = exploredDictionary[destinationNodeId]
    # Walk through the dictionary, building up the path in reverse order
    while previous != None:
        path.append(previous)
        previous = exploredDictionary.get(previous)
    # Reverse the already reversed path, putting it in proper sequential order
    path.reverse()
    return path


def depth_first_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Depth First Search")
    originNodeId = origin[0]
    destinationNodeId = destination[0]

    # Add items to frontier as a tuple in the form (addedByNodeId, NodeThatWasAddedId)
    frontier = Stack()
    explored = []
    pathDictionary = {}

    # Let the origin add itself to the frontier
    frontier.put(originNodeId)

    # Set up some flags to assist twith the search
    isExploring = True
    hasFoundDestination = False
    # Declare a variable to be used below that represents the current node being
    # explored - taken from the frontier.
    currentNodeId = None

    # Timer used to print out percent complete every 3 seconds
    # Used to improve UX during processing
    nextPrintTime = time.time() + 3

    # Start exploring the environment be walking through the nodes along the frontier
    while isExploring:

        # Timer used to print out percent complete every 3 seconds
        # Used to improve UX during processing
        if(time.time() > nextPrintTime):
            percentComplete = len(explored)/(len(frontier) + len(explored))
            print("{00:.2%}".format(percentComplete), " complete")
            nextPrintTime = time.time() + 3

        # If there are no more nodes on the frontier, stop exploring
        if len(frontier) == 0:
            isExploring = False
            break
        # Get the next node on the frontier, and let it be the current node
        previousNodeId = currentNodeId
        currentNodeId = frontier.get()

        # Add the current node to the list of explored nodes
        explored.append(currentNodeId)

        # Add its neighbors to the frontier as necessary
        for connectedNodeId in graph.neighbors(currentNodeId):
            # Do not add to the frontier if already in frontier or already explored
            if connectedNodeId not in frontier and connectedNodeId not in explored:
                # just add it to the frontier
                frontier.put(connectedNodeId)
                # Add the current node to the pathDictionary, noting which node added
                pathDictionary[connectedNodeId] = currentNodeId

                # Determine if this node's id is the destination node's id
                if connectedNodeId == destinationNodeId:
                    hasFoundDestination = True
                    isExploring = False
                    break

    # Work back through the path dictionary to determine the path from the origin
    # to the destination, getting a list of nodeId's back
    path = backtrack(graph, destinationNodeId, pathDictionary)

    return metrics(graph, path, explored)


def breadth_first_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Breadth First Search")
    originNodeId = origin[0]
    destinationNodeId = destination[0]

    # Add items to frontier as a tuple in the form (addedByNodeId, NodeThatWasAddedId)
    frontier = Queue()
    explored = []
    pathDictionary = {}

    # Let the origin add itself to the frontier
    frontier.put(originNodeId)

    # Set up some flags to assist twith the search
    isExploring = True
    hasFoundDestination = False
    # Declare a variable to be used below that represents the current node being
    # explored - taken from the frontier.
    currentNodeId = None

    # Timer used to print out percent complete every 3 seconds
    # Used to improve UX during processing
    nextPrintTime = time.time() + 3

    # Start exploring the environment be walking through the nodes along the frontier
    while isExploring:

        # Timer used to print out percent complete every 3 seconds
        # Used to improve UX during processing
        if(time.time() > nextPrintTime):
            percentComplete = len(explored)/(len(frontier) + len(explored))
            print("{00:.2%}".format(percentComplete), " complete")
            nextPrintTime = time.time() + 3

        # If there are no more nodes on the frontier, stop exploring
        if len(frontier) == 0:
            isExploring = False
            break
        # Get the next node on the frontier, and let it be the current node
        previousNodeId = currentNodeId
        currentNodeId = frontier.get()

        # Add the current node to the list of explored nodes
        explored.append(currentNodeId)

        # Add its neighbors to the frontier as necessary
        for connectedNodeId in graph.neighbors(currentNodeId):
            # Do not add to the frontier if already in frontier or already explored
            if connectedNodeId not in frontier and connectedNodeId not in explored:
                # just add it to the frontier
                frontier.put(connectedNodeId)
                # Add the current node to the pathDictionary, noting which node added
                pathDictionary[connectedNodeId] = currentNodeId

                # Determine if this node's id is the destination node's id
                if connectedNodeId == destinationNodeId:
                    hasFoundDestination = True
                    isExploring = False
                    break

    # Work back through the path dictionary to determine the path from the origin
    # to the destination, getting a list of nodeId's back
    path = backtrack(graph, destinationNodeId, pathDictionary)

    return metrics(graph, path, explored)


def uninformed_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Uniform Cost Search")
    originNodeId = origin[0]
    destinationNodeId = destination[0]

    # Add items to frontier as a tuple in the form (addedByNodeId, NodeThatWasAddedId)
    frontier = PriorityQueue()
    explored = []
    pathDictionary = {}

    # Let the origin add itself to the frontier
    frontier.put((originNodeId, 0), 0)

    # Set up some flags to assist twith the search
    isExploring = True
    hasFoundDestination = False
    # Declare a variable to be used below that represents the current node being
    # explored - taken from the frontier.
    currentNodeId = None

    # Timer used to print out percent complete every 3 seconds
    # Used to improve UX during processing
    nextPrintTime = time.time() + 3

    # Start exploring the environment be walking through the nodes along the frontier
    while isExploring:

        # Timer used to print out percent complete every 3 seconds
        # Used to improve UX during processing
        if(time.time() > nextPrintTime):
            percentComplete = len(explored)/(len(frontier) + len(explored))
            print("{00:.2%}".format(percentComplete), " complete")
            nextPrintTime = time.time() + 3

        # If there are no more nodes on the frontier, stop exploring
        if len(frontier) == 0:
            isExploring = False
            break
        # Get the next node on the frontier, and let it be the current node
        previousNodeId = currentNodeId
        currentNode = frontier.get()
        currentNodeId = currentNode[0]
        currentNodeDist = currentNode[1]

        # Add the current node to the list of explored nodes
        explored.append(currentNodeId)

        # Add its neighbors to the frontier as necessary
        for connectedNodeId in graph.neighbors(currentNodeId):
            # Make a simplified version of the frontier that does not include the distance
            # ie, a list of Id's instead of a list of tuples in the form (id, dist)
            nodesInFrontier = [item[0] for item in frontier]

            # Do not add to the frontier if already in frontier or already explored
            if connectedNodeId not in nodesInFrontier and connectedNodeId not in explored:
                # Keep a running total of the distance travelled up to this point
                distance = currentNodeDist + \
                    getDistance(graph, currentNodeId, connectedNodeId)
                frontier.put((connectedNodeId, distance), distance)
                # Add the current node to the pathDictionary, noting which node added
                pathDictionary[connectedNodeId] = currentNodeId

                # Determine if this node's id is the destination node's id
                if connectedNodeId == destinationNodeId:
                    hasFoundDestination = True
                    isExploring = False
                    break

    # Work back through the path dictionary to determine the path from the origin
    # to the destination, getting a list of nodeId's back
    path = backtrack(graph, destinationNodeId, pathDictionary)

    return metrics(graph, path, explored)


# -- Set up Origin Point
origin_point = (36.30321114344463, -82.36710826765649)  # ETSU
origin = ox.get_nearest_node(G, origin_point)
origin_node = (origin, G.nodes[origin])

destinations = [
    ("Walmart", 36.3089548, -82.4060683),
    ("Target", 36.3426194, -82.3756855),
    ("Tweetsie Trail", 36.3150095, -82.3386371),
    ("Friedberg's", 36.3164826, -82.3547546),
    ("Food City", 36.3018953, -82.3400047),
    ("Best Buy", 36.3479073, -82.4029607)
]

for endLocation in destinations:

    # -- Set up Destination Point
    destination_point = (endLocation[1], endLocation[2])
    destination = ox.get_nearest_node(G, destination_point)
    destination_node = (destination, G.nodes[destination])

    bfs_distance = 0
    dfs_distance = 0
    lat = []
    long = []

    startTime = time.time()
    bfs_route, lat, long, bfs_distance, bfs_steps = breadth_first_search(
        G, origin_node, destination_node)
    route_path = node_list_to_path(G, bfs_route)
    plot_path(lat, long, origin_node, destination_node)
    bfs_runTime = time.time() - startTime

    startTime = time.time()
    dfs_route, lat, long, dfs_distance, dfs_steps = depth_first_search(
        G, origin_node, destination_node)
    route_path = node_list_to_path(G, dfs_route)
    plot_path(lat, long, origin_node, destination_node)
    dfs_runTime = time.time() - startTime

    startTime = time.time()
    ucs_route, lat, long, ucs_distance, ucs_steps = uninformed_search(
        G, origin_node, destination_node)
    route_path = node_list_to_path(G, ucs_route)
    plot_path(lat, long, origin_node, destination_node)
    ucs_runTime = time.time() - startTime

    print(endLocation[0])
    print("Total Route Distance (BFS):", "{0:.3f}miles".format(bfs_distance))
    print("Total Route Distance (DFS):", "{0:.3f}miles".format(dfs_distance))
    print("Total Route Distance (UCS):", "{0:.3f}miles".format(ucs_distance))
    print("")
    print("Total Run Time (BFS):", "{0:.3f}seconds".format(bfs_runTime))
    print("Total Run Time (DFS):", "{0:.3f}seconds".format(dfs_runTime))
    print("Total Run Time (UCS):", "{0:.3f}seconds".format(ucs_runTime))
    print("")
    print("Total Steps (BFS):", bfs_distance)
    print("Total Steps (DFS):", dfs_distance)
    print("Total Steps (UCS):", ucs_distance)
