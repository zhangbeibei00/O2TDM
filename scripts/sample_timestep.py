import numpy as np


def sample_timestep(cand, timestep_num, diffusion_step):
    if isinstance(cand, str):
        cand = eval(cand)
    p_at_splitpoints = cand
    # breakpoint()
    interval = diffusion_step // (len(p_at_splitpoints) - 1)
    split_points = [i * interval for i in range(len(p_at_splitpoints))]
    p = np.interp(range(diffusion_step), split_points, p_at_splitpoints)
    print(p)
    normalized_p = p / np.sum(p)
    indices_np = np.random.choice(len(p), size=(timestep_num,), p=normalized_p)
    # print(indices_np)
    indices_np.sort()
    timelist=list(indices_np)
    return timelist


if __name__ == '__main__':
    timesteps = sample_timestep("[5, 9, 9, 9, 5, 7, 5, 6, 5, 10, 7]", 250, 4000)
    print(timesteps,end=',')
