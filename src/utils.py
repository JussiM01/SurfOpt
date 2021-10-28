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

    return np.array([create_arc(angle, num_steps, start_point, end_point, sign)
                     for angle, sign in angles_signs])

def create_arc(angle, num_steps, start_point, end_point):

    return [arc_point(step, num_steps, angle, start_point, end_point, sign)
            for i in range(num_steps)]


def arc_point(step, num_steps, angle, start_point, end_point, sign):

    if start_point[1] == end_point[1]:
        raise NotImplementedError

    else:
        raise NotImplementedError


def sample_sines(params):

    raise NotImplementedError
