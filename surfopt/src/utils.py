import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def write_config(data, filename):

    config_file = os.path.join('surfopt/config_files',  filename)
    with open(config_file, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_config(filename):

    config_file = os.path.join('surfopt/config_files',  filename)
    with open(config_file, 'r') as conf_file:
        config = json.load(conf_file)

    return config


def unpack(string, mode):

    if mode == 'int':
        return [int(char) for char in string.split(',')]

    elif mode == 'float':
        return [float(char) for char in string.split(',')]


def create_grid(params, surfacemap):

    x = np.linspace(params['x_min'], params['x_max'], params['x_size'])
    y = np.linspace(params['y_min'], params['y_max'], params['y_size'])

    X, Y = np.meshgrid(x.astype('float32'), y.astype('float32'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_tensor = torch.from_numpy(np.stack([X, Y], axis=2)).to(device)
    zs_tensor = surfacemap(grid_tensor)
    Z = zs_tensor.cpu().numpy()

    return X, Y, Z


def create_plot(params):

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=True)

    if params['bound'] is not None:
        ax.set_xlim(-params['bound']['x'], params['bound']['x'])
        ax.set_ylim(-params['bound']['y'], params['bound']['y'])

    ax.grid(False)

    return fig, ax


def sample_sine_sums(params):

    start = params['start']
    end = params['end']
    num_steps = params['num_steps']
    up_ranges = params['up_ranges']
    num_samples = params['num_samples']
    all_sum_params = []
    for i in range(num_samples):
        constants = up_ranges*np.random.random_sample(len(up_ranges))
        all_sum_params.append(zip(constants, params['multiples']))
    sine_sums = np.stack([create_sine_sum(sum_params, start, end, num_steps)
                      for sum_params in all_sum_params], axis=0)
    sine_sums = original_order(sine_sums, start, end)

    return sine_sums


def create_sine_sum(sum_params, start, end, num_steps):

    sines = np.stack([create_sine(
                      const, multiple, [0., -1.], [0., 1.], num_steps)
                      for const, multiple in sum_params], axis=0)
    x_values = np.expand_dims(sines[0, :, 1], axis=1)
    y_values = np.expand_dims(np.sum(sines[:, :, 0], axis=0), axis=1)
    sine_sum = np.concatenate([y_values, x_values], axis=1)
    sine_sum = fix_orientation(sine_sum, start, end)

    return sine_sum


def fix_orientation(sine_sum, start, end):

    fixed_sum = np.apply_along_axis(
        lambda point: fix_point(point, start, end), 1, sine_sum)

    return fixed_sum


def fix_point(point, start, end):

    x_value, y_value = point[0], point[1]
    scale = np.linalg.norm(np.array(end) -np.array(start))
    start, end = reorder(start, end)
    vector = create_vec(x_value, y_value, scale, 1.0, start, end)

    return vector


def sample_line(params):

    params['constants'] = [1.0]
    params['multiples'] = [0]
    line = sample_sines(params)
    
    return line


def sample_sines(params):

    start = params['start']
    end = params['end']
    num_steps = params['num_steps']
    zipped = zip(params['constants'], params['multiples'])
    sines = np.stack([create_sine(const, multiple, start, end, num_steps)
                      for const, multiple in zipped], axis=0)
    sines = original_order(sines, start, end)

    return sines


def create_sine(const, multiple, start, end, num_steps):

    sine = np.stack(
        [sine_point(const, multiple, start, end, step, num_steps)
         for step in range(num_steps)], axis=0)

    return sine


def sine_point(const, multiple, start, end, step, num_steps):

    step_size = 2/(num_steps -1)
    x_value = (step_size*step -1)
    y_value = const*np.sin(multiple*np.pi*x_value)
    scale = np.linalg.norm(np.array(end) -np.array(start))
    start, end = reorder(start, end)
    vector = create_vec(y_value, x_value, scale, 1.0, start, end)

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
    pos_arcs = np.stack([create_arc(angle, num_steps, start, end, 1.0)
                     for angle in angles], axis=0)
    pos_arcs = original_order(pos_arcs, start, end)
    neg_arcs = np.stack([create_arc(angle, num_steps, start, end, -1.0)
                     for angle in angles], axis=0)
    neg_arcs = original_order(neg_arcs, end, start)
    arcs = np.concatenate([pos_arcs, neg_arcs], axis=0)

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


def original_order(trajectories, start, end):

    if (end[0] == start[0]) and (end[1] < start[1]):
        return np.flip(trajectories, axis=1)

    elif end[0] < start[0]:
        return np.flip(trajectories, axis=1)

    return trajectories


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
