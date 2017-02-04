'''
1. Which search algorithm seems to work best for each routing options?
Ans -
A* algorithm works best for all the routing algorithm. It provides optimal solution for all the options
BFS and IDS works also gives optimal solution for segments routing option, but does not give optimal solution for any other option
DFS is not optimal, it works really fast but not optimal(or near optimal)


2. Which algorithm is fastest in terms of the amount of computation time required by your program, and by how much, according to your experiments?
Ans -
    In our experiments, DFS turned out to be a winner in terms of computarional time required
    Algorithm    |        Time Taken to run 50 iteration(Bloomington to Seattle)
    BFS          |      4.40 s
    DFS          |      2.63 s
    IDS          |      Killed after few minutes, obviously the most slowest algorithm in all
    A*           |      12.23 s

    A* is taking a slightly longer time due to handling missing GPS coordinates for some of the cities/junctions, in order to find the optimal solution
    A* checks in a loop for all the cities connected to current city with missing gps and adds all neighbors of the city in fringe.

3. Which algorithm requires the least memory, and by how much, according to your experiments?
Ans -
    IDS would require the least amount of memory, as IDS would find solution certainly at lower depth in the graph than DFS,
    and IDS works in a same manner to DFS, it only holds data till current depth. Hence it would require the least memory
4. Which heuristic function did you use, how good is it, and how might you make it
better?
Ans -
    We have used different heuristics for different routing options: All the options uses havershine distance. Havershine distance
    is distance between two location calculated using their latitude and longitudes.
    Segments - for segments, we divide the segment with minimum size in entire input with the havershine distance from current city
        to destination city. This heuristic would find the
    Distance - we used havershine distance as heuristic for chosing most promising states from state list
    Time - We used havershine time and maximum speed limit in entire city map. By using maximum speed limit and havershine, we can
        calculate minimum time from a city to reach the destination
    Scenic - We calculated havershine distance, and if the speed limit is less than 55 then we deduct current city distance from the havershine
        else we add it as a cost
5. Supposing you start in Bloomington, which city should you travel to if you want to take the longest possible drive (in miles) that is
 still the shortest path to that city? (In other words, which city is furthest from Bloomington?)
Ans - Skagway,_Alaska is the longest possible drive from Bloomington, Indiana. We found it using dijkstra's algorithm.

The abstraction -
State -
We have used state space as information about city, our state contains of -
    [city_name] [distance to reach this current city from source city] [time to reach this current city from source city] [segments to reach this current city from source city]
Successors -
Successors returns all the neighbouring city to input city. We have pruned all visited cities

Initial State -
Start city with all other fields mentioned in state as 0

Goal State -
Goal state gives us total time, total distance, number of segments

Cost Function -
Cost function varies for different routing options,
    For segments it segments traversed,
    For distance, total distance travelled,
    For time, total time taken to reach the city,
    For scenic, distance to current city
'''
import sys
import heapq
import math
from math import radians, cos, sin, asin, sqrt


def read_input():
    global max_speed_limit
    global min_segment_length
    cities = open("road-segments.txt")
    for line in cities.readlines():
        path = line.strip().split(" ")
        if int(path[2]) == 0:
            path[2] = default_distance
        if path[3].isdigit() == False or int(path[3]) == 0:
            path[3] = default_speed_limit
        if path[0] not in city_graph:
            city_graph[path[0]] = [[path[1]] + [path[2]] + [path[3]] + [path[4]]]
        else:
            city_graph[path[0]] += [[path[1]] + [path[2]] + [path[3]] + [path[4]]]

        if path[1] not in city_graph:
            city_graph[path[1]] = [[path[0]] + [path[2]] + [path[3]] + [path[4]]]
        else:
            city_graph[path[1]] += [[path[0]] + [path[2]] + [path[3]] + [path[4]]]

        if int(path[2]) < min_segment_length:
            min_segment_length = int(path[2])
        if int(path[3]) > max_speed_limit:
            max_speed_limit = int(path[3])

    cities.close()
    gps_file = open("city-gps.txt")
    for line in gps_file.readlines():
        data = line.strip().split(" ")
        city_gps[data[0]] = [data[1]] + [data[2]]
    gps_file.close()


def successors(current_node):
    return city_graph[current_node[0]]


def print_path(visited_parent, current_city):
    # Machine readable output
    result_machine_readable = []
    result_human_readable = []
    city_name = destination
    while city_name in visited_parent:
        previous_city = visited_parent[city_name]
        if previous_city[0] != 'start':
            time_min = int(math.ceil(float(previous_city[3]) * 60))
            time_readable = str(time_min/60) + "h " + str(time_min%60) + "m"
            result_human_readable.append("Drive from " + str(previous_city[0]) + " to " + str(city_name) + " on " + str(
                previous_city[4]) + " for " + str(previous_city[1]) + " mile (time: " + time_readable + ")")
        # Machine readable
        result_machine_readable.append(city_name)
        city_name = previous_city[0]

    result_machine_readable.append(str(round(current_city[2], 3)))
    result_machine_readable.append(str(current_city[1]))

    total_time_min = int(math.ceil(float(current_city[2]) * 60))
    total_time_readable = str(total_time_min/60) + "h " + str(total_time_min%60) + "m"
    result_human_readable.append("Your destination is "+str(current_city[1])+" miles away, You will reach you destination in "+ total_time_readable+"\n")

    while result_human_readable:
        print str(result_human_readable.pop())
    print " "
    while result_machine_readable:
        print str(result_machine_readable.pop()) + "",


def solve_bfs(routing_option):
    print "You have chosen BFS as routing option, BFS would only yield optimal solution for 'Segments' routing option"
    print "BFS may run faster, but it requires a lot of memory to store state list"
    print ""
    visited_parent = {}
    fringe = [[start_city] + [0] + [0] + [0]]
    visited_parent[start_city] = ['start', '0', '0', '0', '0']
    while fringe:
        current_city = fringe.pop(0)
        if current_city[0] == destination:
            print_path(visited_parent, current_city)
            return

        for s in successors(current_city):
            if s[0] not in visited_parent:
                fringe.append([s[0]] + [int(current_city[1]) + int(s[1])] + [
                    float(current_city[2]) + (float(s[1]) / float(s[2]))] + [int(current_city[3]) + 1])
                visited_parent[s[0]] = [current_city[0]] + [s[1]] + [s[2]] + [float(s[1]) / int(s[2])] + [s[3]]
    return False


def solve_dfs(routing_option):
    print "You have chosen DFS routing algorithm, DFS is not optimal for any routing option"
    print "Unless you want to travel entire United States before you reach you destination, please use some other algorithm"
    print ""
    visited_parent = {}
    fringe = [[start_city] + [0] + [0] + [0]]
    visited_parent[start_city] = ['start', '0', '0', '0', '0']
    while fringe:
        current_city = fringe.pop()
        if current_city[0] == destination:
            print_path(visited_parent, current_city)
            return

        for s in successors(current_city):
            if s[0] not in visited_parent:
                fringe.append([s[0]] + [int(current_city[1]) + int(s[1])] + [
                    float(current_city[2]) + (float(s[1]) / float(s[2]))] + [int(current_city[3]) + 1])
                visited_parent[s[0]] = [current_city[0]] + [s[1]] + [s[2]] + [float(s[1]) / int(s[2])] + [s[3]]
    return False


def solve_ids(routing_option):
    print "You have chosen IDS as routing algorithm, IDS would only yield optimal solution for 'Segments' routing option"
    print "IDS may take sometime to find the solution, but memory requirements are lower than BFS"
    print ""
    max_level = 0
    while True:
        fringe = [[start_city] + [0] + [0] + [0]]
        visited_parent = {}
        visited_level = {}
        visited_parent[start_city] = ['start', '0', '0', '0', '0']
        visited_level[start_city] = 0
        while fringe:
            current_city = fringe.pop()
            if current_city[0] == destination:
                print_path(visited_parent, current_city)
                return
            if int(current_city[3]) > max_level:
                continue

            for s in successors(current_city):
                if s[0] in visited_parent and (int(visited_level[s[0]]) <= (int(current_city[3]) + 1)):
                    continue
                fringe.append([s[0]] + [int(current_city[1]) + int(s[1])] + [
                    float(current_city[2]) + (float(s[1]) / float(s[2]))] + [int(current_city[3]) + 1])
                visited_parent[s[0]] = [current_city[0]] + [s[1]] + [s[2]] + [float(s[1]) / int(s[2])] + [s[3]]
                visited_level[s[0]] = int(current_city[3]) + 1
        max_level += 1
    print "Failed"


def astar(routing_option):
    options = {
        'segments': f_segments,
        'distance': f_distance,
        'time': f_time,
        'scenic': f_scenic
    }
    fringe = []
    visited_parent = {}
    visited_level = {}
    heapq.heappush(fringe, (1, [start_city] + [0] + [0] + [0]))
    visited_parent[start_city] = ['start', '0', '0', '0', '0']
    visited_level[start_city] = 0
    lat_destination = float(city_gps[destination][0])
    lon_destination = float(city_gps[destination][1])
    while fringe:
        current_city = heapq.heappop(fringe)[1]
        if current_city[0] == destination:
            print_path(visited_parent, current_city)
            return
        for s in successors(current_city):
            if s[0] in visited_parent and (int(visited_level[s[0]]) <= (int(current_city[3]) + 1)):
                continue
            if s[0] in city_gps:
                lat_s = float(city_gps[s[0]][0])
                lon_s = float(city_gps[s[0]][1])
                fx = options[routing_option](current_city, s, lat_s, lon_s, lat_destination, lon_destination)
                heapq.heappush(fringe, (fx, [s[0]] + [int(current_city[1]) + int(s[1])] + [
                    float(current_city[2]) + (float(s[1]) / float(s[2]))] + [int(current_city[3]) + 1]))
                visited_parent[s[0]] = [current_city[0]] + [s[1]] + [s[2]] + [float(s[1]) / int(s[2])] + [s[3]]
                visited_level[s[0]] = int(current_city[3]) + 1
            else:
                city_no_gps = [[s[0]] + [int(current_city[1]) + int(s[1])] + [
                    float(current_city[2]) + (float(s[1]) / float(s[2]))] + [int(current_city[3]) + 1]]
                visited_parent[s[0]] = [current_city[0]] + [s[1]] + [s[2]] + [float(s[1]) / int(s[2])] + [s[3]]
                visited_level[s[0]] = int(current_city[3]) + 1
                while city_no_gps:
                    current_lookahead_city = city_no_gps.pop()
                    for s_lookahead in city_graph[current_lookahead_city[0]]:
                        if s_lookahead[0] in visited_parent and (
                                    int(visited_level[s_lookahead[0]]) <= (int(current_lookahead_city[3]) + 1)):
                            continue
                        if s_lookahead[0] not in city_gps:
                            city_no_gps.append(
                                [s_lookahead[0]] + [int(current_lookahead_city[1]) + int(s_lookahead[1])] + [
                                    float(current_lookahead_city[2]) + (
                                        float(s_lookahead[1]) / float(s_lookahead[2]))] + [
                                    int(current_lookahead_city[3]) + 1])
                            visited_parent[s_lookahead[0]] = [current_lookahead_city[0]] + [s_lookahead[1]] + [
                                s_lookahead[2]] + [float(s_lookahead[1]) / float(s_lookahead[2])] + [s_lookahead[3]]
                            visited_level[s_lookahead[0]] = int(current_lookahead_city[3]) + 1
                            continue
                        lat_s = float(city_gps[s_lookahead[0]][0])
                        lon_s = float(city_gps[s_lookahead[0]][1])
                        fx = options[routing_option](current_lookahead_city, s_lookahead, lat_s, lon_s, lat_destination,
                                                     lon_destination)
                        heapq.heappush(fringe, (
                            fx, [s_lookahead[0]] + [int(current_lookahead_city[1]) + int(s_lookahead[1])] + [
                                float(current_lookahead_city[2]) + (float(s_lookahead[1]) / float(s_lookahead[2]))] + [
                                int(current_lookahead_city[3]) + 1]))
                        visited_parent[s_lookahead[0]] = [current_lookahead_city[0]] + [s_lookahead[1]] + [
                            s_lookahead[2]] + [float(s_lookahead[1]) / float(s_lookahead[2])] + [s_lookahead[3]]
                        visited_level[s_lookahead[0]] = int(current_lookahead_city[3]) + 1
    return False


def f_segments(current_city, s, lat_s, lon_s, lat_destination, lon_destination):
    if (s[0] == destination):
        return int(current_city[3]) + 1
    return int(current_city[3]) + 1 + float(min_segment_length) / haversine(lat_s, lon_s, lat_destination,
                                                                            lon_destination)


def f_distance(current_city, s, lat_s, lon_s, lat_destination, lon_destination):
    return int(current_city[1]) + int(s[1]) + haversine(lat_s, lon_s, lat_destination, lon_destination)


def f_time(current_city, s, lat_s, lon_s, lat_destination, lon_destination):
    return float(current_city[2]) + (float(s[1]) / float(s[2])) + haversine(lat_s, lon_s, lat_destination,
                                                                            lon_destination) / max_speed_limit


def f_scenic(current_city, s, lat_s, lon_s, lat_destination, lon_destination):
    if int(s[2]) < 55: # Not a highway hence subtract current distance from havershine distance
        return int(current_city[1]) + haversine(lat_s, lon_s, lat_destination, lon_destination) - int(s[1])
    else:
        return int(current_city[1]) + int(s[1]) + haversine(lat_s, lon_s, lat_destination, lon_destination)


# This function finds havershine distance between two locations based on latitude and longitude
# The function is referenced from http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
# By - Michael Dunn
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3959  # Radius of earth in miles
    return c * r


# -------------------------------------Execution starts here-------------------------

# Default distance and speed limit, if input does not have those then use this values
default_distance = 50
default_speed_limit = 50

# Max speed limit, used to find the minimum time required from a city to destination(havershine distance/max_speed)
max_speed_limit = 0
min_segment_length = sys.maxint

city_graph = {}
city_gps = {}

read_input()
if len(sys.argv) < 5:
    print "Please pass [start-city] [end-city] [routing-option] [routing-algorithm] in the arguments"
    exit()
start_city = sys.argv[1]
destination = sys.argv[2]

if start_city not in city_graph:
    print "Can not find the city: " + start_city
    exit()
if destination not in city_graph:
    print "Can not find the city: " + destination
    exit()
if start_city == destination:
    print "You are already at you destination"
    exit()
if destination not in city_gps:
    count = 0
    lat = 0.0
    lon = 0.0
    for city in city_graph[destination]:
        if city[0] in city_gps:
            count += 1
            lat += float(city_gps[city[0]][0])
            lon += float(city_gps[city[0]][1])
    if count > 0:
        lat = lat / count
        lon = lon / count

    city_gps[destination] = [lat, lon]

routing_option = sys.argv[3]
routing_algorith = sys.argv[4]

algorithms = {
    'bfs': solve_bfs,
    'dfs': solve_dfs,
    'ids': solve_ids,
    'astar': astar
}


algorithms[routing_algorith](routing_option)

