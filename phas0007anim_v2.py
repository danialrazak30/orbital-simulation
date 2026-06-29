import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def animate_trajectory_2d(trajectory, fps, max_duration=None, title=None, xlabel="x", ylabel="y"):
    """
    Animate the given 2D trajectory.

    The frame rate (fps) must be specified. If no maximum duration is specified then
    the animation will use one step of the trajectories for each frame of the animation.
    This can lead to very long animations, or exhaust the available memory, so if
    a maximum duration is specified then the trajectory will be downsampled (by using
    only every Nth point) to fit within the allowed time.

    Arguments:
        trajectory: trajectory represented as a 2D array [x_values,y_values]
        fps: frames per second
        duration: (optional) duration of animation [s]

    Returns:
        HTML containing JavaScript animation
    """
     # put trajectory in list and delegate
    return animate_trajectories_2d([trajectory], fps, max_duration, title, xlabel, ylabel)


def animate_trajectories_2d(trajectories, fps, max_duration=None, title=None, xlabel="x", ylabel="y"):
    """
    Simultaneously animate the given 2D trajectories.

    The frame rate (fps) must be specified. If no maximum duration is specified then
    the animation will use one step of the trajectories for each frame of the animation.
    This can lead to very long animations, or exhaust the available memory, so if
    a maximum duration is specified then the trajectory will be downsampled (by using
    only every Nth point) to fit within the allowed time.

    Arguments:
        trajectories: list of trajectories, each represented as a 2D array [x_values,y_values]
        fps: frames per second
        max_duration: (optional) max duration of animation [s]
        xlabel, ylabel: labels for x and y axis
    
    Returns:
        HTML containing JavaScript animation
    """
    # Check if we need to turn interactive mode back on afterwards
    was_in_interactive_mode = plt.isinteractive()
    plt.ioff()

    # Combine trajectories into single 3D array, padding trajectories to same length
    trajectories_padded = combine_trajectories(trajectories)
    n_steps = np.shape(trajectories_padded[0])[1]  # All now have same length so use first

    # If we have more steps than needed, downsample data
    if max_duration:
        n_frames_max = max_duration * fps
        trajectories_padded = downsample(trajectories_padded, n_frames_max)
    
    # Get actual number of frames after downsampling, if any
    n_frames = np.shape(trajectories_padded[0])[1]

    # Find coordinate range we need to include in display
    x_min, x_max, y_min, y_max = bounds(trajectories)
    x_min, x_max, y_min, y_max = expand_bounds(x_min, x_max, y_min, y_max, 0.1)

    # Create figure and axes
    fig, axes = plt.subplots()
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_aspect('equal')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if title:
        plt.title(title)

    # Create objects representing bodies and trails
    n_trajectories = len(trajectories)
    bodies, trails = [], []
    for i in range(n_trajectories):
        body,  = axes.plot([],[],'o')           # Line2D with disc markers
        col    = body.get_color()               # get colour of body ...
        trail, = axes.plot([],[],'-',color=col) # and set trail (line) to same colour
        bodies.append(body)         # add Line2D objects to lists
        trails.append(trail)

    # Update plotted points for each frame
    def update(i):
        for j in range(n_trajectories):
            trajectory = trajectories_padded[j]
            body = bodies[j]
            trail = trails[j]
            x,y = trajectory[:,i]
            x_trail, y_trail = trajectory[:,:i+1]
            body.set_data([x],[y])
            trail.set_data(x_trail, y_trail)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, n_frames, interval=1000/fps) # ms per frame

    # Switch interactive mode back on if it was on beforehand
    if was_in_interactive_mode:
        plt.ion()

    # Convert animation to HTML / JavaScript
    ani = HTML(ani.to_jshtml())
    plt.clf() # avoid problems with extra plots appearing in notebook
    return ani


def combine_trajectories(trajectories):
    """
    Pads other trajectories to same length as longest trajectory.

    Arguments:
        trajectories: list of trajectories before padding

    Returns:
        
    """
    # Find largest length out of all trajectories
    traj_lengths = [np.shape(traj)[1] for traj in trajectories]
    max_length = max(traj_lengths)

    # Create empty list ready to append padded trajectories
    padded_trajectories = []

    for traj in trajectories:
        padded_trajectory = pad_trajectory(traj, max_length)
        padded_trajectories.append(padded_trajectory)

    return padded_trajectories


def pad_trajectory(trajectory, length):
    """
    Increase length of trajectory if needed to reach required length, by repeating
    the final point as many times as required.

    Arguments:
        trajectory: trajectory before padding, as 2D array
        length:     required length, which must be no smaller than the existing length
    
    Returns:
        trajectory after padding, as 2D array
    """
    old_length = np.shape(trajectory)[1]
    if length <= old_length:                 # already long enough, nothing to do
        return trajectory
    else:
        x, y = trajectory
        final_x, final_y = x[-1], y[-1]      # final point
        n_extra = length - old_length        # number of points to add
        x_extension = np.array([final_x] * n_extra) # points to add
        y_extension = np.array([final_y] * n_extra)
        new_x = np.append(x, x_extension)    # append new points
        new_y = np.append(y, y_extension)
        return np.array([new_x, new_y] )     # return as 2D array

def bounds(trajectories):
    """
    Get min and max coordinate values for list of trajectories.
    
    Arguments:
        trajectories: list of trajectories as 2D arrays
    
    Return:
        (x_min, x_max, y_min, y_max)
    """

    x_min = min([bounds_single(traj)[0] for traj in trajectories])
    x_max = max([bounds_single(traj)[1] for traj in trajectories])
    y_min = min([bounds_single(traj)[2] for traj in trajectories])
    y_max = max([bounds_single(traj)[3] for traj in trajectories])
    return x_min, x_max, y_min, y_max


def bounds_single(trajectory):
    """
    Get min and max coordinate values for single trajectory.
    
    Arguments:
        trajectory: trajectory before padding, as 2D array
    
    Return:
        (x_min, x_max, y_min, y_max)
    """
    x_min = min([x for x in trajectory[0]])
    x_max = max([x for x in trajectory[0]])
    y_min = min([y for y in trajectory[1]])
    y_max = max([y for y in trajectory[1]])
    return x_min, x_max, y_min, y_max

def expand_bounds(x_min, x_max, y_min, y_max, factor):
    """
    Expand range in x and y by given factor
    """
    x_padding = factor * (x_max - x_min)
    y_padding = factor * (y_max - y_min)
    return x_min - x_padding, x_max + x_padding, y_min - y_padding, y_max + y_padding
              
def downsample(trajectories, n_frames):
    """
    Downsample trajectories (assumed to have equal length) to no more than given number of frames.

    Arguments:
        trajectories: list of trajectories (each as 2xN array)
        n_frames:     maximum desired number of frames
    
    Return:
        list of trajectories downsampled as required
    """
    n_steps = np.shape(trajectories[0])[1]  # All now have same length so use first
    if n_steps <= n_frames:                 # Already short enough
        return trajectories
    ratio = n_steps / n_frames
    skip = int(np.ceil(ratio))              # Downsampling ratio
    trajectories_new = []
    for trajectory in trajectories:
        trajectory_new = trajectory[:,::skip]
        trajectories_new.append(trajectory_new)
    return trajectories_new