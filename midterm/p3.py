#!/usr/bin/env python3
import numpy as np
import cv2
import math 
from queue import PriorityQueue as PQ


class PathPlanning:

    def __init__(self):

        self.load_env()
        self.user_input()

        self.node_queue = PQ()
        self.node_queue.put((0, self.start_point))
        self.current_edge_cost = {self.start_point: 0}
        self.parent = {self.start_point: self.start_point}
        self.visited_matrix = [[[0 for _ in range(12)] for _ in range(1200)] for _ in range(500)]
        self.visited ={}

       # self.astar()
        self.animate()

    def visited_flag(self, node):
        self.visited[node] = True
    
    def visited_check(self, node):
        return node in self.visited
        
    # This fn loads the map with obstacles
    def load_env(self):

        self.width = 1200
        self.height = 500
        brightness = 255
        self.map = np.ones((self.height, self.width, 3), dtype=np.uint8) * brightness

        # list of all rectangle obstacles
        # 'rb' stands for bloated rectangles [bloated by 5 units]
        # 'r' stands for normal rectangles
        rectangle_obstacles = [
            ['rb', (1020, 45),  (1105, 455)], 
            ['rb', (895, 45),  (1105, 130)],  
            ['rb', (895, 370),  (1105, 455)], 
            ['rb', (270, 0), (355, 405)],  
            ['rb', (0, 0),  (1200, 5)], 
            ['rb', (95, 95),  (180, 500)], 
            ['rb', (0, 0),  (5, 500)],  
            ['rb', (1195, 0),  (1200, 500)], 
            ['rb', (0, 495),  (1200, 500)],
            ['r', (900, 375),  (1100, 450)],  
            ['r', (900, 50),  (1100, 125)],  
            ['r', (275, 0), (350, 400)],  
            ['r', (100, 100),  (175, 500)],  
            ['r', (1025, 50),  (1100, 450)] 
        ]

        # list containg hexagonal obstacle
        hexagon_obstacles=[[(650, 250), 150]]

        # add obstacles to map
        for obs in rectangle_obstacles:
        
            corrected_p1 = self.normal_coords_to_cv2(obs[1][0], obs[1][1])
            corrected_p2 = self.normal_coords_to_cv2(obs[2][0], obs[2][1])

            if obs[0] == 'r':
                cv2.rectangle(self.map, corrected_p1, corrected_p2, (70,70,70), -1)

            else:
                cv2.rectangle(self.map, corrected_p1, corrected_p2, (255,90,90), -1)

        for obs in hexagon_obstacles:

            middle_point = self.normal_coords_to_cv2(obs[0][0], obs[0][1])

            side_lines = []
            number_of_lines = 6
            internal_angle = 60
            for line_no in range(0, number_of_lines):
                angle = math.pi*(internal_angle*line_no-30)/180.0
                x_line = middle_point[0] + math.cos(angle)*obs[1]
                y_line = middle_point[1] + math.sin(angle)*obs[1]
                side_lines.append((int(x_line), int(y_line)))

            cv2.fillPoly(self.map, [np.array(side_lines)], (70,70,70))
            cv2.polylines(self.map, [np.array(side_lines)], thickness=4, color=(255,90,90),isClosed=True)

 
    # This fn is used to convert co-ordinates to cv2 co-ordinates for retrival of pixel values
    def normal_coords_to_cv2(self,x_normal, y_normal):
        y_cv2 = (self.height -1)  - y_normal
        x_cv2 = x_normal
        return x_cv2, y_cv2

    # This fn prompts user for input
    def user_input(self):

        self.start_point=None
        self.goal_point=None
        self.start_point_orientation = None
        self.goal_point_orientation = None
        self.step_size=None
        

        while True:

            user_input = input("Please input the start point coordinates [x y theta (either -60,-30,0,30 or 60)] separated by a space: ")
            coords = user_input.split()
            if len(coords)!=3:
                print ("Invaid point! Try again...")
                
            else:
                coords[0]= int(coords[0])
                coords[1]=int(coords[1])
                coords[2] = int(coords[2])
                x, y = self.normal_coords_to_cv2(coords[0], coords[1])

                if x<0 or x>=1200 or y<0 or y>=500 :
                    print ("Invaid point! Try again...")

                elif self.map[y][x][2]!=255:
                    print ("Invaid point! Try again...")
                    
                elif coords[2] not in [-60, -30, 0, 30, 60]:
                    print("Please give the orientation in required format")
                else:
                    start_point=(coords[0],coords[1])
                    start_point_orientation = coords[2]
                    print("Start point details recorded!")
                    break

        while True:

            user_input = input("Please input the goal point coordinates [x y theta (either -60,-30,0,30 or 60)] separated by a space: ")
            coords = user_input.split()
            if len(coords)!=3:
                print ("Invaid point! Try again...")
            else:     
                coords[0]= int(coords[0])
                coords[1]=int(coords[1])
                coords[2]=int(coords[2])
                x, y = self.normal_coords_to_cv2(coords[0], coords[1])
                if x<0 or x>=1200 or y<0 or y>=500 :
                    print ("Invaid point! Try again...")
                elif self.map[y][x][2]!=255:
                    print ("Invaid point! Try again...")
                elif coords[2] not in [-60, -30, 0, 30, 60]:
                    print("Please give the orientation in required format")
                else:
                    goal_point=(coords[0],coords[1])
                    goal_point_orientation = coords[2]
                    print("Goal point details recorded!")
                    break   

        while True:

            user_input = input("Please enter the clearance, robot radius and step size ((1 <= L <= 10) separated by a space: ")
            inputs = user_input.split()
            if len(inputs)!=3:
                print ("Invaid inputs! Try again...")
            else:     
                inputs[0]= int(inputs[0])
                inputs[1]=int(inputs[1])
                inputs[2]=int(inputs[2])
                if inputs[0] > 10 :
                    print ("Very high clearance! Try again...")
                elif inputs[1] > 10 :
                    print ("Very high robot radius! Try again...")
                elif 1<= inputs[2]<=10 :
                    print("Step size is out of range! Try again....")
                else:
                    clearance=inputs[0]
                    robot_radius = inputs[1]
                    step_size = inputs[2]
                    print("Required details recorded!")
                    break   
                
        print ("Start point--> "+ str(start_point) + " " + str(start_point_orientation))     
        print ("Goal point--> "+ str(goal_point)+ " " + str(goal_point_orientation) ) 
        print ("Clearance --> "+ str(clearance))   
        print ("Robot radius --> "+ str(robot_radius)) 
        print ("Step Size --> "+ str(step_size))   
       

        self.start_point = start_point
        self.goal_point = goal_point 
        self.start_point_orientation = start_point_orientation
        self.goal_point_orientation = goal_point_orientation
        self.clearance = clearance
        self.robot_radius = robot_radius
        self.step_size = step_size
        
        print ("Computing path...") 

        
    def move(self, node, degrees):
        new_theta = (node[2] + degrees) % 360
        updated_x = node[0] + self.step_size * math.cos(math.radians(new_theta))
        updated_y = node[1] + self.step_size * math.sin(math.radians(new_theta))
        return updated_x, updated_y, new_theta
    
    def action_set(self,node):
        adjacent_nodes = []
        for degrees in [-60,-30,30,60]:
           adj_node_state = self.move(node,degrees)
           adjacent_nodes.append(adj_node_state)
        return adjacent_nodes
        

    # this fn generates the animation
    def animate(self):
        start_x, start_y = self.normal_coords_to_cv2(self.start_point[0], self.start_point[1])
        goal_x, goal_y = self.normal_coords_to_cv2(self.goal_point[0], self.goal_point[1])
        node_incr = 0

        # # We display the node visit sequence
        # for visited_node in self.visited_list:
        #     node_incr += 1
        #     xn,yn= self.normal_coords_to_cv2(visited_node[0],visited_node[1])
        #     self.map[yn, xn] = (150, 150, 150)

        #     cv2.circle(self.map, (start_x, start_y), 6, (50, 255, 50), -1)  
        #     cv2.circle(self.map, (goal_x, goal_y), 6, (50, 50, 255), -1) 
        #     # To speed up the animation we update frame every 10 nodes
        #     if node_incr % 10 == 0:
        #         cv2.imshow("Map", self.map)
        #         cv2.waitKey(1)
        
        # We display the path
        for node in self.robot_movement:
            xn,yn = self.normal_coords_to_cv2(node[0], node[1])
            cv2.circle(self.map, (xn, yn), 1, (255, 255, 0), -1) 
            cv2.imshow("Map", self.map)
            cv2.waitKey(1)

        cv2.imshow("Map", self.map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def euclidean_distance(self, node1, node2):
        d = math.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)
        return round(d,1)


    # This fn runs the astar algorithm
    def astar(self):
        goal_threshold = 1.5 * self.robot_radius
        infinite = float('inf')
        # Loop until queue has nodes
        while not self.node_queue.empty():

            present_total_cost, node = self.node_queue.get()
            curr_edge_cost  = self.current_edge_cost.get(node, infinite)

            # skip the node if node already optimised
            if present_total_cost > curr_edge_cost:
                continue
            
            # break if goal reached
            if self.euclidean_distance(node, self.goal_point) <= goal_threshold:
                print("GOAL Reached!")
                break
            
            adjacent_nodes = self.action_set(node)
            for adjacent_node in adjacent_nodes:
                adjacent_node_coords = adjacent_node[:2]
                # calculate euclidean distance
                added_edge_cost= self.euclidean_distance(node,adjacent_node_coords)
                updated_edge_cost = self.current_edge_cost[node] + added_edge_cost

                # add/update adjacent node to our system
                if not self.visited_check(adjacent_node_coords) or updated_edge_cost < self.current_edge_cost.get(adjacent_node_coords, float('inf')):
                    self.current_edge_cost[adjacent_node] = updated_edge_cost
                    lowest_edge_cost = updated_edge_cost
                    self.visited_list.append(adjacent_node)
                    self.node_queue.put((lowest_edge_cost, adjacent_node))
                    self.parent[adjacent_node] = node
                    self.visited_flag(adjacent_node_coords)
                    
        # calculate path using backtracking
        self.robot_movement = []
        node = self.goal_point
        while node != self.start_point:
            self.robot_movement.append(node)
            node = self.parent[node]
        self.robot_movement.append(self.start_point)
        self.robot_movement.reverse()



if __name__ == "__main__":
    PathPlanning()