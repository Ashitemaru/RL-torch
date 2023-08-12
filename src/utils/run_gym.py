from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames(frames, path, fps=60):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    anim = animation.FuncAnimation(
        plt.gcf(),
        lambda i: patch.set_data(frames[i]),
        frames=len(frames),
        interval=50,
    )
    anim.save(path, writer="imagemagick", fps=fps)


def run_environment(env, policy, max_steps=100, path="./default.gif", fps=60):
    """
    Params:

    - `env`. Must be a gymnasium environment with render mode `"rgb_array"`
    - `policy`. Have better be an instance of `PolicyWrapper` class
    - `max_steps`. Should be the same with the `TimeLimit` gym wrapper, default to `100`
    - `path`. Where to save the GIF, default to `"./default.gif"`
    - `fps`. Frames per second, which controls the speed of GIF, default to `60`
    """

    episode_reward = 0
    state, _ = env.reset()
    frames = []
    while True:
        frames.append(env.render())
        action = policy.get_action(state)
        state, reward, terminal, truncated, _ = env.step(action)
        episode_reward += reward

        if terminal or truncated:
            break

    frames.append(env.render())
    if not terminal:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

    save_frames(frames, path, fps)


if __name__ == "__main__":
    pass
