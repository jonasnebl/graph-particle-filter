import numpy as np

def extend_perception(robot_perceptions):
    """Extend the perception of the robot with the nodes that are observable.
    """
    
    robot_perceptions_extended = []
    for robot_perception in robot_perceptions:
        robot_position = robot_perception['ego_position']
        perceived_humans = robot_perception['perceived_humans']
        
        # # calculate if a node is observable from the robot's position
        # observability = get_observability(robot_position)

        # assign the observed humans to a node 
        human_node_ids = []
        for human in perceived_humans:
            human_position = human['pos_mean']
            # human_node_ids.append(get_belonging_node(human_position))
            human_node_ids.append(0)
        robot_perceptions_extended.append(human_node_ids)

    return np.array(robot_perceptions_extended, dtype=int)
