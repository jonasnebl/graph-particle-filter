class ParticleTracker:
    def __init__(self):
        self.N_Humans_max = 10
        self.N_particles = 1000

    def add_observation(self, robot_perceptions):
        """Update P based on a set of robot perceptions.
        """
        for robot_perception in robot_perceptions:
            robot_position = robot_perception['ego_position']
            perceived_humans = robot_perception['perceived_humans']
            
            # calculate if a node is observable from the robot's position
            observability = self.get_observability(robot_position)

            # assign the observed humans to a node 
            human_node_ids = []
            for human in perceived_humans:
                human_position = human['pos_mean']
                human_node_ids.append(self.get_belonging_node(human_position))

            # update P based on the nodes that are observable but no human is 
            

    def predict(self):
        pass