'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time
'''

from functools import lru_cache
import time
import math
import argparse

import numpy as np
from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

import onnxruntime as ort
from numba import njit

# decision tree stuff
from sklearn import tree
from sklearn.tree import _tree
from tqdm import tqdm
import pickle

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend

    p = 'bak_matplotlib.mlpstyle'
    plt.style.use(['bmh', p])

def load_network(last_cmd):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''

    onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    #print(f"Loading {mat_filename}...")
    #matfile = loadmat(mat_filename)
    #range_for_scaling = matfile['range_for_scaling'][0]
    #means_for_scaling = matfile['means_for_scaling'][0]
    #mat_filename = f"ACASXU_run2a_1_1_batch_2000.mat"

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    session = ort.InferenceSession(onnx_filename)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session, range_for_scaling, means_for_scaling

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for last_cmd in range(5):
        nets.append(load_network(last_cmd))

    return nets


'Lru Cache.'
def get_time_elapse_mat(command1, dt, command2=0):
    '''get the matrix exponential for the given command

    state: x, y, vx, vy, x2, y2, vx2, vy2
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command1]
    y2 = y_list[command2]

    dtheta1 = (y1 / 180 * np.pi)
    dtheta2 = (y2 / 180 * np.pi)

    a_mat = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0], # x' = vx
        [0, 0, 0, 1, 0, 0, 0, 0], # y' = vy
        [0, 0, 0, -dtheta1, 0, 0, 0, 0], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0, 0, 0, 0, 0], # vy' = vx * dtheta1
    #
        [0, 0, 0, 0, 0, 0, 1, 0], # x' = vx
        [0, 0, 0, 0, 0, 0, 0, 1], # y' = vy
        [0, 0, 0, 0, 0, 0, 0, -dtheta2], # vx' = -vy * dtheta2
        [0, 0, 0, 0, 0, 0, dtheta2, 0], # vy' = vx * dtheta1
        ], dtype=float)

    return expm(a_mat * dt)

def run_network(network_tuple, x, stdout=False):
    'run the network and return the output'

    session, range_for_scaling, means_for_scaling = network_tuple

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})

    return outputs[0][0]

@njit(cache=True)
def state7_to_state5(state7, v_own, v_int):
    """compute rho, theta, psi from state7"""

    assert len(state7) == 7

    x1, y1, theta1, x2, y2, theta2, _ = state7

    rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    dy = y2 - y1
    dx = x2 - x1

    theta = np.arctan2(dy, dx)
    psi = theta2 - theta1

    theta -= theta1

    while theta < -np.pi:
        theta += 2 * np.pi

    while theta > np.pi:
        theta -= 2 * np.pi

    if psi < -np.pi:
        psi += 2 * np.pi

    while psi > np.pi:
        psi -= 2 * np.pi

    return np.array([rho, theta, psi, v_own, v_int])

@njit(cache=True)
def state7_to_state8(state7, v_own, v_int):
    """compute x,y, vx, vy, x2, y2, vx2, vy2 from state7"""

    assert len(state7) == 7

    x1 = state7[0]
    y1 = state7[1]
    vx1 = math.cos(state7[2]) * v_own
    vy1 = math.sin(state7[2]) * v_own

    x2 = state7[3]
    y2 = state7[4]
    vx2 = math.cos(state7[5]) * v_int
    vy2 = math.sin(state7[5]) * v_int

    return np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])

@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""

    img = plt.imread('airplane.png')

    return img

def init_time_elapse_mats(dt):
    """get value of time_elapse_mats array"""

    rv = []

    for cmd in range(5):
        rv.append([])

        for int_cmd in range(5):
            mat = get_time_elapse_mat(cmd, dt, int_cmd)
            rv[-1].append(mat)

    return rv

@njit(cache=True)
def step_state(state7, v_own, v_int, time_elapse_mat, dt):
    """perform one time step with the given commands"""

    state8_vec = state7_to_state8(state7, v_own, v_int)

    s = time_elapse_mat @ state8_vec

    # extract observation (like theta) from state
    new_time = state7[-1] + dt
    theta1 = math.atan2(s[3], s[2])
    theta2 = math.atan2(s[7], s[6])
    rv = np.array([s[0], s[1], theta1, s[4], s[5], theta2, new_time])

    return rv

class State():
    'state of execution container'

    nets = load_networks()
    plane_size = 1500

    nn_update_rate = 1.0 # todo: make this a parameter
    dt = 1.0

    'Valid range" [100, 1145]'
    #v_own = 800 # ft/sec

    'Valid range: [60,1145]'
    #v_int = 500

    time_elapse_mats = init_time_elapse_mats(dt)

    def __init__(self, init_vec, v_own=800, v_int=500, save_states=False):
        assert len(init_vec) == 7, "init vec should have length 7"

        self.vec = np.array(init_vec, dtype=float) # current state
        self.next_nn_update = 0
        self.command = 0 # initial command
        self.v_own = v_own
        self.v_int = v_int

        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.vec_list = [] # state history
        self.commands = [] # commands history
        self.int_commands = [] # intruder command history

        # used only if plotting
        self.artists_dict = {} # set when make_artists is called
        self.img = None # assigned if plotting

        # assigned by simulate()
        self.u_list = []
        self.u_list_index = None
        self.min_dist = np.inf

    def __str__(self):
        x1, y1, _theta1, x2, y2, _theta2, _ = self.vec
        rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        return f'State(v_own: {self.v_own}, v_int: {self.v_int}, rho: {rho})'

    def artists_list(self):
        'return list of artists'

        return list(self.artists_dict.values())

    def set_plane_visible(self, vis):
        'set ownship plane visibility status'

        self.artists_dict['dot0'].set_visible(not vis)
        self.artists_dict['circle0'].set_visible(False) # circle always False
        self.artists_dict['lc0'].set_visible(True)
        self.artists_dict['plane0'].set_visible(vis)

    def update_artists(self, axes):
        '''update artists in self.artists_dict to be consistant with self.vec, returns a list of artists'''

        assert self.artists_dict
        rv = []

        x1, y1, theta1, x2, y2, theta2, _ = self.vec

        for i, x, y, theta in zip([0, 1], [x1, x2], [y1, y2], [theta1, theta2]):
            key = f'plane{i}'

            if key in self.artists_dict:
                plane = self.artists_dict[key]
                rv.append(plane)

                if plane.get_visible():
                    theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                    original_size = list(self.img.shape)
                    img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
                    rotated_size = list(img_rotated.shape)
                    ratios = [r / o for r, o in zip(rotated_size, original_size)]
                    plane.set_data(img_rotated)

                    size = State.plane_size
                    width = size * ratios[0]
                    height = size * ratios[1]
                    box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                    tbox = TransformedBbox(box, axes.transData)
                    plane.bbox = tbox

            key = f'dot{i}'
            if key in self.artists_dict:
                dot = self.artists_dict[f'dot{i}']
                cir = self.artists_dict[f'circle{i}']
                rv += [dot, cir]

                dot.set_data([x], [y])
                cir.set_center((x, y))

        # line collection
        lc = self.artists_dict['lc0']
        rv.append(lc)

        int_lc = self.artists_dict['int_lc0']
        rv.append(int_lc)

        self.update_lc_artists(lc, int_lc)

        return rv

    def update_lc_artists(self, own_lc, int_lc):
        'update line collection artist based on current state'

        assert self.vec_list

        for lc_index, lc in enumerate([own_lc, int_lc]):
            paths = lc.get_paths()
            colors = []
            lws = []
            paths.clear()
            last_command = -1
            codes = []
            verts = []

            for i, vec in enumerate(self.vec_list):
                if np.linalg.norm(vec - self.vec) < 1e-6:
                    # done
                    break

                if lc_index == 0:
                    cmd = self.commands[i]
                else:
                    cmd = self.int_commands[i]

                x = 0 if lc_index == 0 else 3
                y = 1 if lc_index == 0 else 4

                # command[i] is the line from i to (i+1)
                if cmd != last_command:
                    if codes:
                        paths.append(Path(verts, codes))

                    codes = [Path.MOVETO]
                    verts = [(vec[x], vec[y])]

                    if cmd == 1: # weak left
                        lws.append(2)
                        colors.append('b')
                    elif cmd == 2: # weak right
                        lws.append(2)
                        colors.append('c')
                    elif cmd == 3: # strong left
                        lws.append(2)
                        colors.append('g')
                    elif cmd == 4: # strong right
                        lws.append(2)
                        colors.append('r')
                    else:
                        assert cmd == 0 # coc
                        lws.append(2)
                        colors.append('k')

                codes.append(Path.LINETO)

                verts.append((self.vec_list[i+1][x], self.vec_list[i+1][y]))

            # add last one
            if codes:
                paths.append(Path(verts, codes))

            lc.set_lw(lws)
            lc.set_color(colors)

    def make_artists(self, axes, show_intruder):
        'make self.artists_dict'

        assert self.vec_list
        self.img = get_airplane_img()

        posa_list = [(v[0], v[1], v[2]) for v in self.vec_list]
        posb_list = [(v[3], v[4], v[5]) for v in self.vec_list]

        pos_lists = [posa_list, posb_list]

        if show_intruder:
            pos_lists.append(posb_list)

        for i, pos_list in enumerate(pos_lists):
            x, y, theta = pos_list[0]

            l = axes.plot(*zip(*pos_list), f'c-', lw=0, zorder=1)[0]
            l.set_visible(False)
            self.artists_dict[f'line{i}'] = l

            if i == 0:
                lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(lc)
                self.artists_dict[f'lc{i}'] = lc

                int_lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(int_lc)
                self.artists_dict[f'int_lc{i}'] = int_lc

            # only sim_index = 0 gets intruder aircraft
            if i == 0 or (i == 1 and show_intruder):
                size = State.plane_size
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, axes.transData)
                box_image = BboxImage(tbox, zorder=2)

                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)

                box_image.set_data(img_rotated)
                axes.add_artist(box_image)
                self.artists_dict[f'plane{i}'] = box_image

            if i == 0:
                dot = axes.plot([x], [y], 'k.', markersize=6.0, zorder=2)[0]
                self.artists_dict[f'dot{i}'] = dot

                rad = 1500
                c = patches.Ellipse((x, y), rad, rad, color='k', lw=3.0, fill=False)
                axes.add_patch(c)
                self.artists_dict[f'circle{i}'] = c

    def step(self):
        'execute one time step and update the model'

        tol = 1e-6

        if self.next_nn_update < tol:
            assert abs(self.next_nn_update) < tol, f"time step doesn't sync with nn update time. " + \
                      f"next update: {self.next_nn_update}"

            # update command
            self.update_command()

            self.next_nn_update = State.nn_update_rate

        self.next_nn_update -= State.dt
        intruder_cmd = self.u_list[self.u_list_index]

        if self.save_states:
            self.commands.append(self.command)
            self.int_commands.append(intruder_cmd)

        time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd] #get_time_elapse_mat(self.command, State.dt, intruder_cmd)

        self.vec = step_state(self.vec, self.v_own, self.v_int, time_elapse_mat, State.dt)

    def simulate(self, cmd_list):
        '''simulate system

        saves result in self.vec_list
        also saves self.min_dist
        '''

        self.u_list = cmd_list
        self.u_list_index = None

        assert isinstance(cmd_list, list)
        tmax = len(cmd_list) * State.nn_update_rate

        t = 0.0

        if self.save_states:
            rv = [self.vec.copy()]

        #self.min_dist = 0, math.sqrt((self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2), self.vec.copy()
        prev_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2

        while t + 1e-6 < tmax:
            self.step()

            cur_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2

            if self.save_states:
                rv.append(self.vec.copy())

            t += State.dt

            if cur_dist_sq > prev_dist_sq:
                #print(f"Distance was increasing at time {round(t, 2)}, stopping simulation. Min_dist: {round(prev_dist, 1)}ft")
                break

            prev_dist_sq = cur_dist_sq

        self.min_dist = math.sqrt(prev_dist_sq)

        if self.save_states:
            self.vec_list = rv

        if not self.save_states:
            assert not self.vec_list
            assert not self.commands
            assert not self.int_commands

    def update_command(self):
        'update command based on current state'''

        rho, theta, psi, v_own, v_int = state7_to_state5(self.vec, self.v_own, self.v_int)

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

        if rho > 60760:
            self.command = 0
        else:
            last_command = self.command

            net = State.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = run_network(net, state)
            self.command = np.argmin(res)

            #names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        if self.u_list_index is None:
            self.u_list_index = 0
        else:
            self.u_list_index += 1

            # repeat last command if no more commands
            self.u_list_index = min(self.u_list_index, len(self.u_list) - 1)

def plot(s, save_mp4):
    """plot a specific simulation"""

    s.vec = s.vec_list[0] # for printing the correct state
    print(f"plotting state {s}")

    init_plot()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axes.axis('equal')

    axes.set_title("ACAS Xu Simulations")
    axes.set_xlabel('X Position (ft)')
    axes.set_ylabel('Y Position (ft)')

    time_text = axes.text(0.02, 0.98, 'Time: 0', horizontalalignment='left', fontsize=14,
                          verticalalignment='top', transform=axes.transAxes)
    time_text.set_visible(True)

    custom_lines = [Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='c', lw=2),
                    Line2D([0], [0], color='r', lw=2)]

    axes.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear of Conflict', 'Weak Right', 'Strong Right'], \
                fontsize=14, loc='lower left')

    s.make_artists(axes, show_intruder=True)
    states = [s]

    plt.tight_layout()

    num_steps = len(states[0].vec_list)
    interval = 20 # ms per frame
    freeze_frames = 10 if not save_mp4 else 80

    num_runs = 1 # 3
    num_frames = num_runs * num_steps + 2 * num_runs * freeze_frames

    #plt.savefig('plot.png')
    #plot_commands(states[0])

    def animate(f):
        'animate function'

        if not save_mp4:
            f *= 5 # multiplier to make animation faster

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        run_index = f // (num_steps + 2 * freeze_frames)

        f = f - run_index * (num_steps + 2*freeze_frames)

        f -= freeze_frames

        f = max(0, f)
        f = min(f, num_steps - 1)

        num_states = len(states)

        if f == 0:
            # initiaze current run_index
            show_plane = num_states <= 10
            for s in states[:num_states]:
                s.set_plane_visible(show_plane)

            for s in states[num_states:]:
                for a in s.artists_list():
                    a.set_visible(False)

        time_text.set_text(f'Time: {f * State.dt:.1f}')

        artists = [time_text]

        for s in states[:num_states]:
            s.vec = s.vec_list[f]
            artists += s.update_artists(axes)

        for s in states[num_states:]:
            artists += s.artists_list()

        return artists

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)

    if save_mp4:
        writer = animation.writers['ffmpeg'](fps=50, metadata=dict(artist='Stanley Bak'), bitrate=1800)

        my_anim.save('sim.mp4', writer=writer)
    else:
        plt.show()

def make_random_input(seed, intruder_can_turn=True, num_inputs=100):
    """make a random input for the system"""

    np.random.seed(seed) # deterministic random numbers

    # state vector is: x, y, theta, x2, y2, theta2, time
    init_vec = np.zeros(7)
    init_vec[2] = np.pi / 2 # ownship moving up initially

    radius = 10000 + np.random.random() * 55000 # [10000, 65000]
    angle = np.random.random() * 2 * np.pi
    int_x = radius * np.cos(angle)
    int_y = radius * np.sin(angle)
    int_heading = np.random.random() * 2 * np.pi

    init_vec[3] = int_x
    init_vec[4] = int_y
    init_vec[5] = int_heading

    # intruder commands for every control period (0 to 4)
    if intruder_can_turn:
        cmd_list = []

        for _ in range(num_inputs):
            cmd_list.append(np.random.randint(5))
    else:
        cmd_list = [0] * num_inputs

    # generate random valid velocities
    #init_velo = [np.random.randint(100, 1146),
    #             np.random.randint(60, 1146)]
    init_velo = [np.random.randint(100, 1200),
                 np.random.randint(0, 1200)]

    return init_vec, cmd_list, init_velo

def main():
    'main entry point'

    # parse arguments
    parser = argparse.ArgumentParser(description='Run ACASXU Dublins model simulator.')
    parser.add_argument("--save-mp4", action='store_true', default=False, help="Save plotted mp4 files to disk.")
    parser.add_argument("--intruder-turn", action='store_true', default=False, help="Toggles boolean flag to allow intruder to perform \
                                                                                     commands other than flying straight.")
    parser.add_argument("--fixed-seed", type=int, default=None, help="Simulates the parameters generated by provided seed.")
    args = parser.parse_args()

    intruder_can_turn = args.intruder_turn
    save_mp4 = args.save_mp4
    fixed_seed = args.fixed_seed

    interesting_seed = -1
    interesting_state = None

    if fixed_seed is not None:
        interesting_seed = fixed_seed
    else:
        num_sims = 10000
        # with 10000 sims, seed 671 has min_dist 4254.5ft

        start = time.perf_counter()

        for seed in range(num_sims):
            if seed % 1000 == 0:
                print(f"{(seed//1000) % 10}", end='', flush=True)
            elif seed % 100 == 0:
                print(".", end='', flush=True)

            init_vec, cmd_list, init_velo = make_random_input(seed, intruder_can_turn=intruder_can_turn)

            v_own = init_velo[0]
            v_int = init_velo[1]

            # reject start states where initial command is not clear-of-conflict
            state5 = state7_to_state5(init_vec, v_own, v_int)

            if state5[0] > 60760:
                command = 0 # rho exceeds network limit
            else:
                res = run_network(State.nets[0], state5)
                command = np.argmin(res)

            if command != 0:
                continue

            # run the simulation
            s = State(init_vec, v_own, v_int, save_states=True)
            s.simulate(cmd_list)

            # save most interesting state based on some criteria
            if interesting_state is None or s.min_dist < interesting_state.min_dist:
                interesting_seed = seed
                interesting_state = s

        diff = time.perf_counter() - start
        ms_per_sim = round(1000 * diff / num_sims, 3)
        print(f"\nDid {num_sims} sims in {round(diff, 1)} secs ({ms_per_sim}ms per sim)")

    # optional: do plot
    assert interesting_seed != -1

    init_vec, cmd_list, init_velo = make_random_input(interesting_seed, intruder_can_turn=intruder_can_turn)
    s = State(init_vec, init_velo[0], init_velo[1], save_states=True)
    s.simulate(cmd_list)

    print(len(s.vec_list))
    print(s.commands)

    d = round(s.min_dist, 2)
    print(f"\nSeed {interesting_seed} has min_dist {d}ft")
    plot(s, save_mp4)

# Decision tree stuff
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    print("def predict({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            #print("{}else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
            print("{}if {} > {}".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, np.argmax(tree_.value[node])))

    recurse(0, 1)


if __name__ == "__main__":
    #main()

    nets = load_networks()

    # 0: rho, distance
    # 1: theta, angle to intruder relative to ownship heading
    # 2: psi, heading of intruder relative to ownship heading
    # 3: v_own, speed of ownship
    # 4: v_int, speed in intruder

    # min inputs: 0, -3.1415, -3.1415, 100, 0
    # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

    '''
    for i in range(0, 5):
        net = nets[i]
    '''
    net = nets[0]

    # Training data
    num_rhos = 30
    num_thetas = 10
    num_psis = 10
    num_vOwns = 20
    num_vInts = 20
    '''
    num_rhos = 2
    num_thetas = 2
    num_psis = 2
    num_vOwns = 2
    num_vInts = 2'''
    
    num_nets = 1 # Constant, only have 5 nets we can

    rho_range = np.linspace(0, 60760, num_rhos)
    theta_range = np.linspace(-np.pi, np.pi, num_thetas)
    psi_range = np.linspace(-np.pi, np.pi, num_psis)
    v_own_range = np.linspace(100, 1200, num_vOwns)
    v_int_range = np.linspace(0, 1200, num_vInts)
    nets_range = np.arange(0, num_nets)

    stored_states = np.zeros([num_rhos * num_thetas * num_psis * num_vOwns * num_vInts * num_nets, 5])
    command_nums = np.zeros([num_rhos * num_thetas * num_psis * num_vOwns * num_vInts * num_nets])

    index = 0
    for rho_ind in range(0, num_rhos):
        for theta_ind in range(0, num_thetas):
            for psi_ind in range(0, num_psis):
                for v_own_ind in range(0, num_vOwns):
                    for v_int_ind in range(0, num_vInts):

                        rho = rho_range[rho_ind]
                        theta = theta_range[theta_ind]
                        psi = psi_range[psi_ind]
                        v_own = v_own_range[v_own_ind]
                        v_int = v_int_range[v_int_ind]

                        state = [rho, theta, psi, v_own, v_int]
                        stored_states[index, :] = state.copy()
                        
                        res = run_network(net, state)
                        command = np.argmin(res)

                        command_nums[index] = command

                        index += 1
    #print("Training states")
    #print(stored_states)
    # Test data (randomly generated)
    test_pts = 10000
    test_states = np.random.rand(test_pts, 5)
    test_cmds = np.zeros([test_pts])
    # Scale the te
    for i in range(test_states.shape[0]):
        test_state = test_states[i]
        
        # rescale the test state
        test_state = np.multiply(test_state, np.array([60760, 2 * np.pi, 2 * np.pi, 1100, 1200])) + np.array([0, -np.pi, -np.pi, 100, 0])
        test_states[i] = np.copy(test_state)
        
        test_res = run_network(net, test_state)
        test_cmd = np.argmin(test_res)
        
        
        test_cmds[i] = test_cmd
        
    #print(f'cmd nums: {command_nums.shape}')
    #print("Test states")
    #print(test_states)
    #print(command_nums)


    #print(f'\n\nstates:  {stored_states.shape}')
    #print(stored_states)
    clf = tree.DecisionTreeClassifier()
    #clf.fit(stored_states, command_nums)
    
    # Check out the alpha value thresholds for the tree
    path_tree = clf.cost_complexity_pruning_path(stored_states, command_nums)
    ccp_alphas, impurities = path_tree.ccp_alphas, path_tree.impurities
    
    
    
    #print(f'alphas:  {ccp_alphas}')
    #print(f'impurities:  {impurities}')
    
    # Plot impurity vs alpha value for the tree
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    #plt.savefig('impurity.png')
    
    # Train many trees with different alpha values
    clfs = []
    
    #ccp_alphas = [0, 0.0004, 0.004, 0.04]
    num_nodes = []
    
    print("Iterate alphas")
    for ccp_alpha in tqdm(ccp_alphas):
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, class_weight='balanced')
        clf.fit(stored_states, command_nums)
        num_nodes.append(clf.tree_.node_count)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    # Compare training and testing accuracy of the pruned trees   
    print("Training scores") 
    train_scores = [clf.score(stored_states, command_nums) for clf in tqdm(clfs)]
    print("Testing scores")
    test_scores = [clf.score(test_states, test_cmds) for clf in tqdm(clfs)]
    
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("num nodes")
    ax.set_title("Number of tree nodes vs alpha")
    ax.plot(ccp_alphas, num_nodes, marker='o', drawstyle='steps-post')

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    #plt.savefig('accuracy.png')
    #plt.show()
    
    pickle.dump(ccp_alphas, open('alphas.pickle', 'wb'))
    pickle.dump(train_scores, open('trainScores.pickle', 'wb'))
    pickle.dump(test_scores, open('testScores.pickle', 'wb'))
    pickle.dump(impurities, open('impurities.pickle', 'wb'))
    pickle.dump(num_nodes, open('num_nodes.pickle', 'wb'))
    #pickle.dump(clf, open('best_tree.pickle', 'wb'))
    
    # DON'T SAVE THE TREES, THIS FILE WILL BLOW UP
    #pickle.dump(clfs, open('trees.pickle', 'wb'))
'''
    tree.plot_tree(clf)



    text_representation = tree.export_text(clf)
    #print(text_representation)

    # with open("decision_tree.txt", "w") as fout:
    #     fout.write(text_representation)
    tree_to_code(clf, ['rho', 'theta', 'psi', 'vOwn', 'vInt'])
'''
    #print(len(stored_states.tolist()))
    #print(len(stored_states[0].tolist()))
    #print(len(command_nums.tolist()))



    
