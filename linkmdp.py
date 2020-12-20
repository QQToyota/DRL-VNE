import gym
from gym import spaces
import copy
import numpy as np
from network import Network


class LinkEnv(gym.Env):

    def __init__(self, sub):
        self.count = -1
        self.linkpath = Network.getallpath(sub)
        self.n_action = len(self.linkpath)
        self.sub = copy.deepcopy(sub)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, 2), dtype=np.float32)
        self.state = None
        self.vnr = None
        self.link = None
        btn = Network.getbtns(sub)
        self.btn = []
        self.btn = (btn - np.min(btn)) / (np.max(btn) - np.min(btn))
        self.mbw_remain=[]
        for paths in self.linkpath.values():
            path = list(paths.values())[0]
            self.mbw_remain.append(Network.get_path_capacity(self.sub, path))

    def set_sub(self, sub):
        self.sub = copy.deepcopy(sub)

    def set_vnr(self, vnr):
        self.vnr = vnr

    def set_link(self,link):
        self.link=link

    def step(self, action):
        thepath = list(self.linkpath[action].values())[0]

        path_remain_bw=[]
        i = 0
        while i < len(thepath) - 1:
            fr = thepath[i]
            to = thepath[i + 1]
            self.sub[fr][to]['bw_remain'] -= self.vnr[self.link[0]][self.link[1]]['bw']
            path_remain_bw.append(self.sub[fr][to]['bw_remain'])
            i += 1

        self.mbw_remain[action]=min(path_remain_bw)
        mbw_remain = (self.mbw_remain - np.min(self.mbw_remain)) / (
                np.max(self.mbw_remain) - np.min(self.mbw_remain))

        self.state = (mbw_remain, self.btn)

        return np.vstack(self.state).transpose(), 0.0, False, {}

    def reset(self):
        """获得底层网络当前最新的状态"""
        self.count = -1
        self.actions = []
        mbw = []
        for paths in self.linkpath.values():
            path = list(paths.values())[0]
            mbw.append(Network.get_path_capacity(self.sub, path))

        # normalization
        mbw = (mbw - np.min(mbw)) / (np.max(mbw) - np.min(mbw))
        mbw_remain = mbw

        self.state = (mbw_remain, self.btn)
        return np.vstack(self.state).transpose()

    def render(self, mode='human'):
        pass

