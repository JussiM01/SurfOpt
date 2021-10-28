import matplotlib.pyplot as plt
import numpy as np


def create_countours(params):

    raise NotImplementedError


def create_plot(params):

    raise NotImplementedError


def sample_arcs(params):

    num_angles = params['num_angles']
    min_angle = params['min_angle']
    max_angle = params['max_angle']
    angle_diff = (max_angle - min_angle)/num_angles
    angles = [min_angle + i*angle_diff for i in range(num_angles + 1)]
    num_steps = params['num_steps']
    angles_signs = [(angle, sign) for angle in angles for sign in [-1.0, 1.0]]

    arcs = np.stack([create_arc(angle, num_steps, start_point, end_point, sign)
                     for angle, sign in angles_signs], axis=0)

    return arcs


def create_arc(angle, num_steps, start, end):

    arc = np.stack(
        [arc_point(step, num_steps, angle, start, end, sign)
         for i in range(num_steps)], axis=0)

    return arc


def arc_point(step, num_steps, angle, start, end, sign):

    step_size = angle/(num_steps -1)
    alpha = step_size*step -angle*0.5
    radius = 1/np.sin(angle)
    x_value = radius*(np.cos(alpha) - np.cos(angle))
    y_value = radius*np.sin(alpha)

    if start[0] == end[0]:
        vector = sign*np.array([x_value, y_value])

    else:
        angle_from_x_axis = np.arctan((end[0] -start[0])/(end[1] -start[1]))
        vector = rotate(-angle_from_x_axis, sign*np.array([y_value, x_value]))

    return vector


def rotate(alpha, point):
    rotation_matrix = np.array(
        [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)])

    return rotation_matrix.dot(point)


def sample_sines(params):

    raise NotImplementedError
