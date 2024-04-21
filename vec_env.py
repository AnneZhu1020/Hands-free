import numpy as np
from multiprocessing import Process, Pipe

USE_GRAPH_INFO = False

def worker(remote, parent_remote, env):
    parent_remote.close()
    env.create()
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if USE_GRAPH_INFO:
                    if done:
                        ob, info, graph_info = env.reset()
                        reward = 0
                        done = False
                    else:
                        ob, reward, done, info, graph_info = env.step(data)
                    remote.send((ob, reward, done, info, graph_info))
                else:
                    if done:
                        ob, info = env.reset()
                        reward = 0
                        done = False
                    else:
                        ob, reward, done, info = env.step(data)
                    remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                if USE_GRAPH_INFO:
                    ob, info, graph_info = env.reset()
                    remote.send((ob, info, graph_info))
                else:
                    ob, info = env.reset()
                    remote.send((ob, info))
            elif cmd == 'close':
                env.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, env):
        self.closed = False
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if USE_GRAPH_INFO:
            obs, rewards, dones, infos, graph_infos = zip(*results)
            return np.stack(obs), np.stack(rewards), np.stack(dones), infos, graph_infos
        else:
            obs, rewards, dones, infos = zip(*results)
            return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        if USE_GRAPH_INFO:
            obs, infos, graph_infos = zip(*results)
            return np.stack(obs), infos, graph_infos
        else:
            obs, infos = zip(*results)
            return np.stack(obs), infos

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
