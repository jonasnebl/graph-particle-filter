import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt

class Confidence:
    def __init__(self, sim_log):
        self.sim_log = sim_log

    def per_node(self):
        pass

    def per_graph(self):
        return self.per_node().mean()


class Accuracy:
    def __init__(self, sim_log):
        """Kullback-Leibler divergence."""
        self.sim_log = sim_log
        self.N_trapezoids = 100
        self.bandwidth = 0.1

    def per_node(self):
        # get the kernel density estimation for every node
        klds = []
        x = np.linspace(0, 1, self.N_trapezoids+1)

        test_data = np.array([0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1])
        test_data = np.concatenate(([np.linspace(0,1,test_data.size)], [test_data]), axis=0).T
        print(test_data)

        for node in range(1):
            kde = FFTKDE(kernel='gaussian', bw=0.1).fit(test_data)
                         
            # evalute kde on a grid and plot the result
            # x = np.linspace(-0.1, 1.1, 5)
            # X,Y = np.meshgrid(x,x)
            # flattened_grid_points = (np.array([X.ravel(), Y.ravel()]).T)[np.argsort(X.ravel())]
            # print(flattened_grid_points)

            grid_points = 40

            grid, points = kde.evaluate(grid_points)

            # The grid is of shape (obs, dims), points are of shape (obs, 1)
            x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            z = points.reshape(grid_points, grid_points).T

            # Plot the kernel density estimate
            N = 20
            # plt.contour(x, y, z, N, linewidths=0.8, colors='k')
            plt.contourf(x, y, z, N, cmap="RdBu_r")
            # plt.plot(data[:, 0], data[:, 1], 'ok', ms=3)
            plt.tight_layout()
            plt.show()

            
        return
    
    def per_graph(self):
        return self.per_node().mean()
    
if __name__ == '__main__':
    kld = Accuracy('dummy')
    kld.per_node()
