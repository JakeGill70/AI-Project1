from graphics import *
from collections import deque
import math
from operator import itemgetter
import numpy as np
import time

# Define Colors
red = color_rgb(255, 0, 0)
green = color_rgb(0, 255, 0)
blue = color_rgb(0, 0, 255)
etsu_blue = color_rgb(4, 30, 66)
black = color_rgb(0, 0, 0)
white = color_rgb(255, 255, 255)
gray = color_rgb(162, 170, 173)
etsu_gold = color_rgb(255, 199, 44)


# ==============================================================================
class PriorityQueue:
    """
        Simple Priority Queue Class using a Python list
    """

    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return self.queue.__iter__()

    def put(self, item, priority):
        '''
            Add the item and sort by priority
        '''
        node = [item, priority]
        self.queue.append(node)
        self.queue.sort(key=itemgetter(1))

    def get(self):
        '''
            Return the highest-priority item in the queue
        '''
        if len(self.queue) == 0:
            return None
        node = self.queue.pop(0)
        return node[0]

    def empty(self):
        '''
            Return True if the queue has no items
        '''
        return len(self.queue) == 0

# ==============================================================================


class Queue:
    def __init__(self):
        self.queue = deque()

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

# ==============================================================================


class Stack:
    def __init__(self):
        self.stack = deque()

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


# ==============================================================================
class Field:
    """
        Class Field uses the graphics.py library

        It simulates an x by y field that contains Polygons, Lines, and Points.

        Search Space: Vertexes of each polygon, Start & End locations
    """

    def __init__(self, width, height, intitle):
        '''
            Create lists of points, polygons
        '''
        self.points = []
        self.path = []
        self.polygons = []
        self.extras = []
        self.width = width
        self.height = height
        self.start = Point(0, 0)
        self.end = Point(0, 0)
        self.win = GraphWin(title=intitle, width=width, height=height)

    def setCoords(self, x1, y1, x2, y2):
        '''
            Set the viewport of the Field
        '''
        self.win.setCoords(x1, y1, x2, y2)

    def setBackground(self, color):
        '''
            Set the background color
        '''
        self.win.setBackground(color)

    def add_polygon(self, polygon):
        '''
            Add the polygon and the vertexes of the Polygon
        '''
        start = None
        self.points = self.points + polygon.getPoints()
        start = polygon.getPoints()[0]
        self.polygons.append(polygon)

        polygon.draw(self.win)

    def add_start(self, start):
        '''
            Add and display the starting location
        '''
        self.points.append(start)
        self.start = start
        c = Circle(start, 10)
        c.setFill('green')
        self.extras.append(c)
        text = Text(Point(start.x+15, start.y+25), 'Start')
        text.setSize(20)
        text.setTextColor('gray')
        self.extras.append(text)
        text.draw(self.win)
        c.draw(self.win)

    def add_end(self, end):
        '''
            Add and display the ending location
        '''
        self.points.append(end)
        self.end = end
        c = Circle(end, 10)
        c.setFill('red')
        self.extras.append(c)
        text = Text(Point(end.x+15, end.y-25), 'End')
        text.setSize(20)
        text.setTextColor('gray')
        self.extras.append(text)
        text.draw(self.win)
        c.draw(self.win)

    def get_neighbors(self, node):
        '''
          Returns a list of neighbors of node -- Vertexes that the node can see.
          All vertexes are within node's line-of-sight.
        '''
        neighbors = []

        # Loop through vertexes (stored in self.points)
        for point in self.points:
            # Ignore the vertex if it is the same as the node passed
            if (point == node):
                continue

            intersects = False

            # Create a line that represents a potential path segment
            pathSegment = Line(node, point)

            # Loop through the Polygons in the Field
            for o in self.polygons:
                # If the path segment intersects the Polygon, ignore it.
                if (o.intersects(pathSegment)):
                    intersects = True
                    break

            # If the path segment does not intersect the Polygon, it is a
            #  valid neighbor.
            if (not intersects):
                neighbors.append(point)

        return neighbors

    def wait(self):
        '''
            Pause the Window for action
        '''
        self.win.getMouse()

    def close(self):
        '''
            Closes the Window after a pause
        '''
        self.win.getMouse()
        self.win.close()

    def reset(self, sp=None, ep=None):
        self.path = []
        for extra in self.extras:
            extra.undraw()
        self.extras = []
        if sp is not None:
            self.add_start(sp)
        if ep is not None:
            self.add_end(ep)

    def backtrack(self, came_from, node):
        '''
            Recreate the path located.

            Requires a came_from dictionary that contains the parents of each node.
            The node passed is the end of the path.
        '''
        current = node
        self.path.append(current)
        parent = came_from[str(current)]
        while parent != self.start:
            line = Line(current, parent)
            line.setWidth(4)
            line.setOutline("white")
            line.setArrow("first")
            self.extras.append(line)
            line.draw(self.win)
            current = parent
            parent = came_from[str(current)]
            self.path.append(current)
        line = Line(current, parent)
        line.setWidth(4)
        line.setOutline("white")
        line.setArrow("first")
        line.draw(self.win)
        self.extras.append(line)
        self.path.append(parent)
        self.path.reverse()

    def metrics(self, startTime, explored, path):
        # Calculate the run time
        runTime = time.time() - startTime
        # Count the number of steps - ie nodes checked before a path was found
        steps = len(explored)
        # Calculate the path cost
        pathCost = 0
        for i in range(len(path)-1):
            pathCost += self.straight_line_distance(
                path[i], path[i+1])
        # Return these stats
        return f"(Run Time: {round(runTime, 3)}, Path Cost: {round(pathCost, 3)}, Steps: {round(steps, 3)})"

    def straight_line_distance(self, point1, point2):
        '''
            Returns the straight-line distance between point 1 and point 2
        '''
        sld = math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
        return sld

    def depth_first_search(self):
        '''
            The Depth-First Search Algorithm
        '''
        print("Depth-First Search")
        startTime = time.time()
        frontier = Stack()
        explored = []
        came_from = {}

        # Let the origin add itself to the frontier
        frontier.put(self.start)

        # Set up some flags to assist twith the search
        isExploring = True
        hasFoundEnd = False

        # Declare a variable to be used below that represents the current node being
        # explored - taken from the frontier.
        currentPoint = None

        # Start exploring the environment be walking through the points along the frontier
        while isExploring:
            # If there are no more points on the frontier, stop exploring
            if len(frontier) == 0:
                isExploring = False
                break
            # Get the next node on the frontier, and let it be the current node
            previousPoint = currentPoint
            currentPoint = frontier.get()

            # Add the current node to the list of explored points
            explored.append(currentPoint)

            # Add its neighbors to the frontier as necessary
            for neighbor in self.get_neighbors(currentPoint):
                # Do not add to the frontier if already in the frontier or already explored
                if neighbor not in frontier and neighbor not in explored:
                    # Add it to the frontier
                    frontier.put(neighbor)
                    # Add the neighbor to the came_from dictionary
                    came_from[str(neighbor)] = currentPoint

                    # Determine if this point is the end point
                    if neighbor == self.end:
                        hasFoundEnd = True
                        isExploring = False
                        break

        self.backtrack(came_from, self.end)

        return self.metrics(startTime, explored, self.path)

    def breadth_first_search(self):
        '''
            The Breadth-First Search Algorithm
        '''
        print("Breadth-First Search")
        startTime = time.time()
        frontier = Queue()
        explored = []
        came_from = {}

        # Let the origin add itself to the frontier
        frontier.put(self.start)

        # Set up some flags to assist twith the search
        isExploring = True
        hasFoundEnd = False

        # Declare a variable to be used below that represents the current node being
        # explored - taken from the frontier.
        currentPoint = None

        # Start exploring the environment be walking through the points along the frontier
        while isExploring:
            # If there are no more points on the frontier, stop exploring
            if len(frontier) == 0:
                isExploring = False
                break
            # Get the next node on the frontier, and let it be the current node
            previousPoint = currentPoint
            currentPoint = frontier.get()

            # Add the current node to the list of explored points
            explored.append(currentPoint)

            # Add its neighbors to the frontier as necessary
            for neighbor in self.get_neighbors(currentPoint):
                # Do not add to the frontier if already in the frontier or already explored
                if neighbor not in frontier and neighbor not in explored:
                    # Add it to the frontier
                    frontier.put(neighbor)
                    # Add the neighbor to the came_from dictionary
                    came_from[str(neighbor)] = currentPoint

                    # Determine if this point is the end point
                    if neighbor == self.end:
                        hasFoundEnd = True
                        isExploring = False
                        break

        self.backtrack(came_from, self.end)

        return self.metrics(startTime, explored, self.path)

    def best_first_search(self):
        '''
           The Best-First Search Algorithm

            Uses the Backtrack method to draw the final path when your
             algorithm locates the end point
        '''
        print("Best-First Search")
        startTime = time.time()
        frontier = PriorityQueue()
        explored = []
        came_from = {}

        # Let the origin add itself to the frontier
        frontier.put(self.start, 0)

        # Set up some flags to assist twith the search
        isExploring = True
        hasFoundEnd = False

        # Declare a variable to be used below that represents the current node being
        # explored - taken from the frontier.
        currentPoint = None

        # Start exploring the environment be walking through the points along the frontier
        while isExploring:
            # If there are no more points on the frontier, stop exploring
            if len(frontier) == 0:
                isExploring = False
                break
            # Get the next node on the frontier, and let it be the current node
            previousPoint = currentPoint
            currentPoint = frontier.get()

            # Add the current node to the list of explored points
            explored.append(currentPoint)

            # Add its neighbors to the frontier as necessary
            for neighbor in self.get_neighbors(currentPoint):
                # Convert the frontier from a PQ of tuples to a simple list of Points
                tmpFrontier = [x[0] for x in frontier]
                # Do not add to the frontier if already in the frontier or already explored
                if neighbor not in frontier and neighbor not in explored:
                    # Add it to the frontier
                    frontier.put(neighbor, self.straight_line_distance(
                        neighbor, self.end))
                    # Add the neighbor to the came_from dictionary
                    came_from[str(neighbor)] = currentPoint

                    # Determine if this point is the end point
                    if neighbor == self.end:
                        hasFoundEnd = True
                        isExploring = False
                        break

        self.backtrack(came_from, self.end)

        return self.metrics(startTime, explored, self.path)

    def astar_search(self):
        '''
           The A* Search Algorithm

            Uses the Backtrack method to draw the final path when your
             algorithm locates the end point
        '''
        print("A* Search")
        startTime = time.time()
        # Items in the priority queue should be a tuple in the format (Point, Score)
        # where score is defined as the path cost up to that point plus the heuristic cost.
        frontier = PriorityQueue()
        explored = []
        came_from = {}

        # Let the origin add itself to the frontier
        frontier.put((self.start, 0), 0)

        # Set up some flags to assist twith the search
        isExploring = True
        hasFoundEnd = False

        # Declare a variable to be used below that represents the current node being
        # explored - taken from the frontier.
        currentPoint = None

        # Start exploring the environment be walking through the points along the frontier
        while isExploring:
            # If there are no more points on the frontier, stop exploring
            if len(frontier) == 0:
                isExploring = False
                break
            # Get the next node on the frontier, and let it be the current node
            previousPoint = currentPoint
            currentPointTuple = frontier.get()
            currentPoint = currentPointTuple[0]
            currentDist = currentPointTuple[1]

            # Add the current node to the list of explored points
            explored.append(currentPoint)

            # Add its neighbors to the frontier as necessary
            for neighbor in self.get_neighbors(currentPoint):
                # Convert the frontier from a PQ of tuples to a simple list of Points
                tmpFrontier = [x[0][0] for x in frontier]
                # Do not add to the frontier if already in the frontier or already explored
                if neighbor not in tmpFrontier and neighbor not in explored:
                    # Get the distance from this neighboring point to the end point
                    hueristicDist = self.straight_line_distance(
                        neighbor, self.end)
                    # Get the distance from this neighboring point to the current point
                    neighborDist = self.straight_line_distance(
                        neighbor, currentPoint)
                    # Calculate the path distance from the start point up to this neighboring point
                    pathDist = currentDist + neighborDist

                    # Add the point and the path distance to the frontier
                    # Use the path distance combined with the heuristic to determine its priority
                    frontier.put((neighbor, pathDist),
                                 pathDist + hueristicDist)
                    # Add the neighbor to the came_from dictionary
                    came_from[str(neighbor)] = currentPoint

                    # Determine if this point is the end point
                    if neighbor == self.end:
                        hasFoundEnd = True
                        isExploring = False
                        break

        self.backtrack(came_from, self.end)

        return self.metrics(startTime, explored, self.path)


# ==============================================================================
# ==============================================================================
def setup_game_map(f):
    p0 = Polygon(Point(182, 794), Point(199, 827), Point(234, 838), Point(241, 858), Point(234, 866), Point(263, 896), Point(275, 915), Point(299, 906), Point(321, 927), Point(353, 919), Point(359, 898), Point(394, 893), Point(401, 872), Point(433, 856), Point(460, 862), Point(465, 882), Point(494, 884), Point(512, 866), Point(538, 873), Point(539, 904), Point(556, 924), Point(579, 915), Point(591, 890), Point(625, 886), Point(646, 900), Point(633, 926), Point(646, 945), Point(675, 948), Point(697, 916), Point(725, 920), Point(750, 900), Point(752, 877), Point(784, 860), Point(810, 864), Point(833, 840), Point(828, 816), Point(805, 811), Point(790, 782), Point(809, 759), Point(803, 736), Point(775, 718), Point(775, 688), Point(807, 674), Point(807, 642), Point(832, 628), Point(836, 596), Point(819, 575), Point(819, 548), Point(787, 539), Point(768, 521), Point(773, 497), Point(747, 474), Point(758, 447), Point(738, 427), Point(736, 404), Point(755, 389), Point(740, 353), Point(753, 333), Point(751, 289), Point(782, 274), Point(802, 244), Point(791, 224), Point(820, 200), Point(854, 195), Point(865, 218), Point(887, 224), Point(915, 207), Point(919, 181), Point(949, 168), Point(952, 140), Point(927, 123), Point(933, 93), Point(888, 65), Point(
        871, 85), Point(828, 79), Point(799, 56), Point(807, 34), Point(794, 16), Point(768, 19), Point(731, 28), Point(715, 43), Point(710, 82), Point(684, 100), Point(666, 85), Point(632, 118), Point(652, 146), Point(631, 174), Point(593, 174), Point(571, 198), Point(582, 229), Point(577, 260), Point(556, 284), Point(521, 273), Point(499, 291), Point(474, 273), Point(450, 284), Point(442, 305), Point(456, 324), Point(410, 356), Point(375, 344), Point(377, 309), Point(335, 308), Point(315, 289), Point(286, 284), Point(299, 253), Point(277, 213), Point(258, 201), Point(234, 214), Point(198, 226), Point(170, 228), Point(149, 203), Point(98, 193), Point(91, 152), Point(74, 147), Point(53, 166), Point(29, 205), Point(18, 244), Point(33, 262), Point(31, 305), Point(20, 328), Point(70, 357), Point(68, 383), Point(35, 390), Point(19, 432), Point(19, 489), Point(22, 547), Point(12, 584), Point(21, 615), Point(31, 639), Point(81, 638), Point(117, 595), Point(148, 558), Point(179, 529), Point(211, 536), Point(224, 552), Point(281, 538), Point(324, 519), Point(370, 495), Point(429, 481), Point(464, 492), Point(471, 534), Point(460, 581), Point(441, 611), Point(383, 623), Point(352, 638), Point(316, 672), Point(271, 706), Point(232, 746), Point(197, 773))
    p0.setOutline("tan")
    p0.setWidth(2)
    p0.setFill("green")
    f.add_polygon(p0)


# ==============================================================================
# ==============================================================================
def setup_logo_map(f):
    '''
        Set up the polygons that make up the logo obstacles in this program
    '''
    p1 = Polygon(Point(330, 47), Point(306, 108), Point(345, 120), Point(376, 141),
                 Point(391, 166), Point(391, 285), Point(
                     565, 285), Point(565, 131),
                 Point(781, 131), Point(820, 143), Point(
                     853, 168), Point(895, 238),
                 Point(962, 238), Point(893, 31), Point(
                     874, 41), Point(857, 45),
                 Point(834, 47))
    p1.setFill(etsu_gold)
    p2 = Polygon(Point(339, 327), Point(305, 406), Point(689, 406), Point(715, 376),
                 Point(755, 346), Point(789, 331), Point(
                     795, 312), Point(391, 312),
                 Point(391, 326))
    p2.setFill(etsu_gold)
    p3 = Polygon(Point(391, 431), Point(391, 551), Point(379, 577), Point(356, 597),
                 Point(334, 608), Point(306, 615), Point(
                     330, 673), Point(820, 673),
                 Point(850, 676), Point(875, 682), Point(
                     904, 693), Point(976, 487),
                 Point(908, 487), Point(891, 516), Point(
                     870, 542), Point(839, 567),
                 Point(816, 578), Point(793, 585), Point(
                     765, 589), Point(565, 588),
                 Point(565, 431))
    p3.setFill(etsu_gold)

    text = Text(Point(650, 1000), 'TM')
    text.setSize(20)
    text.setTextColor(etsu_gold)
    text.draw(f.win)

    f.add_polygon(p1)
    f.add_polygon(p2)
    f.add_polygon(p3)


# ==============================================================================
# ==============================================================================
def setup_polygon_field(f):

    # Add other polygons here

    #p4 = Polygon(Point(30,125),Point(200,225),Point(50,20))
    # p4.setFill(gray)
    # f.add_polygon(p4)
    print("My Very Own Polygon Field that I created Myself.")


# ==============================================================================
# ==============================================================================
def main():
    # === Regular Field
    f = Field(1280, 720, "Bucky's Treasure Hunt")
    f.setCoords(0, 720, 1280, 0)
    f.setBackground(etsu_blue)
    setup_logo_map(f)
    starting_point = Point(20, 375)
    ending_point = Point(1200, 700)

    # === Game Map Field
    # f = Field(1024, 1024, "Bucky's Treasure Hunt")
    # f.setCoords(0, 1024, 1024, 0)
    # f.setBackground(etsu_blue)
    # setup_game(f)
    # starting_point = Point(200,100)
    # ending_point = Point(400,600)

    f.add_start(starting_point)
    f.add_end(ending_point)

    f.wait()

    print("Breadth-First Search:", f.breadth_first_search())
    f.wait()
    f.reset(starting_point, ending_point)
    print("A* Search:", f.astar_search())
    f.wait()
    f.reset(starting_point, ending_point)
    print("Depth-First Search:", f.depth_first_search())
    f.wait()
    f.reset(starting_point, ending_point)
    print("Best-First Search:", f.best_first_search())

    f.close()


main()
