import numpy as np
import geoplotlib
import pandas as pd
import pyglet
np.set_printoptions(precision=4,linewidth=1000,suppress=True)
from math import *
def terrace_H():
    H0 = np.mat([[-1.6688907435, -6.9502305710, 940.69592392565],  # terrace1-c0
                 [1.1984806153, -10.7495778320, 868.29873467315],
                 [0.0004069210, -0.0209324057, 0.42949125235]])

    H1 = np.mat([[0.6174778372, -0.4836875683, 147.00510919005],  # terrace2-c1
                 [0.5798503075, 3.8204849039, -386.096405131],
                 [0.0000000001, 0.0077222239, -0.01593391935]])

    H2 = np.mat([[-0.2717592338, 1.0286363982, -17.6643219215],  # terrace2-c2
                 [-0.1373600672, -0.3326731339, 161.0109069274],
                 [0.0000600052, 0.0030858398, -0.04195162855]])

    H3 = np.mat([[-0.3286861858, 0.1142963200, 130.25528281945],  # terrace2-c3
                 [0.1809954834, -0.2059386455, 125.0260427323],
                 [0.0000693641, 0.0040168154, -0.08284534995]])
    H = [H0, H1, H2, H3]
    return H

def img_to_world(img_Points, H):
    input = np.mat([img_Points[0], img_Points[1], 1])
#    input = np.mat([img_Points[0][0][0], img_Points[0][0][1], 1])
    world_points = H * input.T
    world_points = world_points/world_points[2]
    return world_points
if __name__ == '__main__':
    #  H * X_image = X_topview
#    x_image1 = np.array([269, 210], dtype=np.float32).reshape([1, 1, -1])
#    x_image2 = np.array([272, 135], dtype=np.float32).reshape([1, 1, -1])
#    x_image3 = np.array([97, 135], dtype=np.float32).reshape([1, 1, -1])
#    x_image4 = np.array([118, 129], dtype=np.float32).reshape([1, 1, -1])
    x_image1 = np.array([269, 210], dtype=np.float32)
    x_image2 = np.array([272, 135], dtype=np.float32)
    x_image3 = np.array([97, 135], dtype=np.float32)
    x_image4 = np.array([118, 129], dtype=np.float32)
    # m is number of camera, 0,1,2,3
    x_topview1 = img_to_world(x_image1, terrace_H()[0])
    x_topview2 = img_to_world(x_image2, terrace_H()[1])
    x_topview3 = img_to_world(x_image3, terrace_H()[2])
    x_topview4 = img_to_world(x_image4, terrace_H()[3])
    m =1
#    data = pd.read_csv(r'data/bus.csv')
#    geoplotlib.dot(data)
#    geoplotlib.show()

