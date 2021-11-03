import matplotlib.pyplot as plt
import numpy as np


def create_countours(params):

    raise NotImplementedError


def create_plot(params):

    raise NotImplementedError


def sample_sines(params):

    zipped = list(zip(params['constants'], params['multiples']))
    sines = np.stack([create_sine(const, multiple, start, end, num_steps)
                      for const, multiple in zipped], axis=0)

    return sines


def create_sine(const, multiple, start, end, num_steps):

    sine = np.stack(
        [sine_point(const, multiple, start, end, step, num_steps)
         for step in range(num_steps)], axis=0)

    return sine


def sine_point(const, multiple, start, end, step, num_steps):

    step_size = 2/(num_steps -1)
    x_value = np.pi*(step_size*step -1)
    y_value = np.sin(x_value)
    scale = np.linalg.norm(np.array(end) -np.array(start))
    start, end = reorder(start, end)
    vector = create_vec(x_value, y_value, scale, sign, start, end)

    return vector


def sample_arcs(params):

    num_angles = params['num_angles']
    min_angle = params['min_angle']
    max_angle = params['max_angle']
    start = params['start']
    end = params['end']
    angle_diff = (max_angle - min_angle)/num_angles
    angles = [min_angle + i*angle_diff for i in range(num_angles + 1)]
    num_steps = params['num_steps']
    angles_signs = [(angle, sign) for angle in angles for sign in [-1.0, 1.0]]
    arcs = np.stack([create_arc(angle, num_steps, start, end, sign)
                     for angle, sign in angles_signs], axis=0)

    return arcs


def create_arc(angle, num_steps, start, end, sign):

    arc = np.stack(
        [arc_point(step, num_steps, angle, start, end, sign)
         for step in range(num_steps)], axis=0)

    return arc


def arc_point(step, num_steps, angle, start, end, sign):

    step_size = 2*angle/(num_steps -1)
    alpha = step_size*step -angle
    radius = 1/np.sin(angle)
    x_value = radius*(np.cos(alpha) - np.cos(angle))
    y_value = radius*np.sin(alpha)
    scale = np.linalg.norm(np.array(end) -np.array(start))
    start, end = reorder(start, end)
    vector = create_vec(x_value, y_value, scale, sign, start, end)

    return vector


def reorder(start, end):

    if (end[0] == start[0]) and (end[1] < start[1]):
        return end, start

    elif end[0] < start[0]:
        return end, start

    return start, end


def create_vec(x_value, y_value, scale, sign, start, end):

    if start[0] == end[0]:
        vector = sign*np.array([x_value, y_value])
        vector = vector -np.array([0.0, -1.0])
        vector = 0.5*scale*vector
        vector = vector + np.array(start)

    else:
        vector = sign*np.array([y_value, x_value])
        vector = vector -np.array([-1.0, 0.0])
        vector = 0.5*scale*vector
        angle_from_x_axis = np.arctan((end[1] -start[1])/(end[0] -start[0]))
        vector = rotate(angle_from_x_axis, vector)
        vector = vector + np.array(start)

    return vector


def rotate(alpha, point):

    rotation_matrix = np.array(
        [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    return rotation_matrix.dot(point)


def sample_sines(params):

    raise NotImplementedError
