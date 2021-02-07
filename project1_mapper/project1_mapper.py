import networkx as nx
import osmnx as ox
from osmnx import distance as distance
import plotly.graph_objects as go
import collections
import numpy as np
import matplotlib.pyplot as plt

ox.config(log_console=True, use_cache=True)

G = ox.graph_from_address('1600 Pennsylvania Avenue, Washington, DC, USA', dist=3000, network_type='drive')

### Use this code to display a plot of the graph if desired. Note: You need to import matplotlib.pyplot as plt
fig, ax = ox.plot_graph(G, edge_linewidth=3, node_size=0, show=False, close=False)
plt.show()


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


def backtrack(graph, origin, node, explored):
    '''
    Accepts the graph, the origin and the node reached. Also accepts the list of
    explored/visited locations.

    It should work backwards to the origin from node, using explored, so you
    know exactly which path you took to locate the destination.

    You should return a list of osmids in the order of the path.
    It may also be helpful to track the latitudes and longitudes of each item on
    the path so you can easily move it to the plot function, along with the
    path cost (i.e., distance).
    '''
    print("Backtracking")



def depth_first_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Depth First Search")


def breadth_first_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Breadth First Search")


def uninformed_search(graph, origin, destination):
        '''
        Accepts the graph and the origin and destination points
        Returns the result of backtracking through the explored list when the
         destination is found.
        '''
        print("My Uninformed Search Algorithm")


## -- Set up Destination Point
destination_point = (38.889730, -77.005928) # The U.S. Capitol
origin_point = (38.890219, -77.049560) # The Lincoln Memorial
origin = ox.get_nearest_node(G, origin_point)
destination = ox.get_nearest_node(G, destination_point)
origin_node = (origin, G.nodes[origin])
destination_node = (destination, G.nodes[destination])
bfs_distance = 0
dfs_distance = 0
lat = []
long = []

# bfs_route, lat, long, bfs_distance = breadth_first_search(G, origin_node, destination_node)
# route_path = node_list_to_path(G, bfs_route)
# plot_path(lat, long, origin_node, destination_node)

# dfs_route, lat, long, dfs_distance = depth_first_search(G, origin_node, destination_node)
# route_path = node_list_to_path(G, dfs_route)
plot_path(lat, long, origin_node, destination_node) # Until filled in with values, this doesn't do much.

print("Total Route Distance (BFS):", bfs_distance)
print("Total Route Distance (DFS):", dfs_distance)


# The following is example code to save your map to an HTML file.
# route = nx.shortest_path(G, origin_node, destination_node)
# route_map = ox.plot_route_folium(G, route)
# filepath = 'data/graph.html'
# route_map.save(filepath)
# print(G.nodes(True))
# ec = ox.plot.get_edge_colors_by_attr(G, attr='length', cmap='plasma_r')
# fig, ax = ox.plot_graph(G, node_color='w', node_edgecolor='k', node_size=20,
#                         edge_color=ec, edge_linewidth=2)
