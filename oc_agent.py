import torch
import numpy as np
import pickle
import time
import itertools
from os.path import join as pjoin
import jericho
import sentencepiece as spm

from utils import *
from logger import *
from network import *
from replay import *
from glove.glove_utils import *
from network_BIDAF import *

class BaseAgent:
    def __init__(self, config):
        self.config = config
        # self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0


    def close(self):
        close_obj(self.task)

    # def save(self, filename):
    #     torch.save(self.network.state_dict(), '%s.model' % (filename))
    #     with open('%s.stats' % (filename), 'wb') as f:
    #         pickle.dump(self.config.state_normalizer.state_dict(), f)


    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info, graph_info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)


class OptionCriticAgent(BaseAgent):
    def __init__(self, args, envs, vocab=None): #, env):
        BaseAgent.__init__(self, args)
        self.args = args
        self.start = time.time()
        self.envs = envs

        if args.use_graph_info:
            self.state_dim = args.hidden_dim * 3 + 100
        else:
            self.state_dim = args.hidden_dim * 3
        self.action_dim = 1
        word_emb = None

        # obs, self.infos = self.envs.reset()
        # self.states = self.build_state(obs, self.infos)
        # self.states, _ = self.staternn(self.states)  # states: size([8, 384])
        # self.valid_ids = [self.encode(info['valid']) for info in self.infos]  # ['open mailbox', 'north', 'south', 'west']
        self.save_path = args.output_dir
        self.use_graph_info = args.use_graph_info
        self.use_bidaf = args.use_bidaf
        self.state_all = args.state_all

        if self.use_bidaf:
            self.word2id, word_emb = get_dict_emb(args.glove_file)

        self.network = OptionCriticNet(args, FCBody(self.state_dim), self.action_dim, word_emb)

        self.target_network = OptionCriticNet(args, FCBody(self.state_dim), self.action_dim, word_emb)
        self.target_network.load_state_dict(self.network.state_dict())

        self.is_initial_states = tensor(np.ones((args.num_envs))).byte()
        self.prev_options = self.is_initial_states.clone().long()
        self.total_steps = 0
        self.worker_index = tensor(np.arange(args.num_envs)).long()

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)

        if self.use_graph_info:
            self.state_gat = StateNetwork(args.gat_emb_size, vocab, args.statenetwork_emb_size,
                                          args.dropout_ratio, args.tsv_file)
            # self.target_state_gat = StateNetwork(args.gat_emb_size, vocab, args.statenetwork_emb_size,
            #                               args.dropout_ratio, args.tsv_file)
            # self.target_state_gat.load_state_dict(self.state_gat.state_dict())

            self.optimizer = torch.optim.Adam(list(self.network.parameters()) +
                                                 list(self.state_gat.parameters()), args.learning_rate)
        else:
            # self.optimizer = torch.optim.RMSprop(self.network.parameters(), args.learning_rate)
            self.optimizer = torch.optim.Adam(self.network.parameters(), args.learning_rate)

        self.noisy_reward = args.noisy_reward
        if self.noisy_reward:
            self.noisy_value = args.noisy_value
            self.eps = 1e-6

        self.sent_transformer = args.sent_transformer
        self.word_transformer = args.word_transformer
        self.trans_network = args.trans_network

        # self.storage_save = Storage(args.rollout_length,
        #                             ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])

    def build_state(self, obs, infos):
        """ Returns a state representation built from various info sources. """
        obs_ids = [self.sp.EncodeAsIds(o) for o in obs]
        look_ids = [self.sp.EncodeAsIds(info['look']) for info in infos]
        inv_ids = [self.sp.EncodeAsIds(info['inv']) for info in infos]
        return [State(ob, lk, inv) for ob, lk, inv in zip(obs_ids, look_ids, inv_ids)]

    def encode(self, obs_list):
        """ Encode a list of observations """
        return [self.sp.EncodeAsIds(o) for o in obs_list]

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        with torch.no_grad():
            q_option = prediction['q']  # size([8, 3])
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option) + prob
            pi_option.scatter_(1, greedy_option, prob)

            mask = torch.zeros_like(q_option)
            mask[self.worker_index, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option

            dist = torch.distributions.Categorical(probs=pi_option)
            options = dist.sample()
            dist = torch.distributions.Categorical(probs=pi_hat_option)
            option_hat = dist.sample()

            options = torch.where(is_intial_states, options, option_hat)
        return options

    def step(self, tb):
        args = self.args
        storage = Storage(args.rollout_length,
                          ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])
        for _ in range(args.rollout_length):
            self.valid_ids = [self.encode(info['valid']) for info in self.infos]

            prediction = self.network(self.states)  # q, beta
            epsilon = args.random_option_prob(args.num_envs)
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            act_sizes = [len(a) for a in self.valid_ids]
            admissible_cmd_ids = list(itertools.chain.from_iterable(self.valid_ids))  # len: 32
            act_out = self.staternn.forward_enc_act(admissible_cmd_ids)  # size([32, 128])

            pred_pi = self.network.forward_pi(act_out, act_sizes)
            pi = [p[o] for p, o in zip(pred_pi['pi'], options)]
            log_pi = [p[o] for p, o in zip(pred_pi['log_pi'], options)]

            # todo: here need split out? [for i in ]
            actions, entropy = [], []
            for p in pi:
                dist = torch.distributions.Categorical(probs=p)
                actions.append(dist.sample())
                entropy.append(dist.entropy())
            actions = torch.stack(actions)
            entropy = torch.stack(entropy)

            action_strs = [info['valid'][idx] for info, idx in zip(self.infos, actions)]
            obs, rewards, dones, self.infos, graph_info = self.envs.step(action_strs)

            for done, info in zip(dones, self.infos):
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])

            next_states = self.build_state(obs, self.infos)
            next_states, _ = self.staternn(next_states)
            storage.feed(prediction)
            storage.feed({'pi': pi,
                          'log_pi': log_pi,
                          'reward': tensor(rewards).unsqueeze(-1),
                          'option': options.unsqueeze(-1),
                          'prev_option': self.prev_options.unsqueeze(-1),
                          'entropy': entropy.unsqueeze(-1),
                          'action': actions.unsqueeze(-1),
                          'init_state': self.is_initial_states.unsqueeze(-1),
                          'eps': epsilon,
                          'mask': tensor(1 - dones).unsqueeze(-1)})

            self.is_initial_states = tensor(dones).byte()
            self.prev_options = options
            self.states = next_states

            self.total_steps += args.num_envs
            if self.total_steps // args.num_envs % args.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
                # self.target_state_gat.load_state_dict(self.state_gat.state_dict())

        with torch.no_grad():
            prediction = self.target_network(self.states)  # q, beta
            storage.placeholder()
            # betas= prediction['beta'][torch.arange(args.num_envs), self.prev_options] # size([8, 3]) -> size([8, 1])
            # q = prediction['q'][torch.arange(args.num_envs), self.prev_options] # size([8, 3]) -> size([8, 1])

            betas = prediction['beta'][self.worker_index, self.prev_options]  # size([8, 3]) -> size([8, 1])
            q = prediction['q'][self.worker_index, self.prev_options]  # size([8, 3]) -> size([8, 1])

            ret = (1 - betas) * q + betas * torch.max(prediction['q'], dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(args.rollout_length)):
            ret = storage.reward[i] + args.discount * storage.mask[i] * ret
            if self.adv_variant:
                adv = ret - torch.max(storage.q[i])
            else:
                adv = ret - storage.q[i].gather(1, storage.option[i])

            storage.ret[i] = ret
            storage.advantage[i] = adv

            v = (1 - storage.eps[i]) * storage.q[i].max(dim=-1, keepdim=True)[0] + \
                storage.eps[i] * storage.q[i].mean(-1).unsqueeze(-1)
            q = storage.q[i].gather(1, storage.prev_option[i])
            storage.beta_advantage[i] = q - v + args.termination_regularizer

        entries = storage.extract(['beta', 'log_pi', 'action', 'entropy', 'ret', 'advantage', 'beta_advantage',
                                   'init_state', 'q', 'option', 'prev_option'])
        log_pi = entries.log_pi
        action = entries.action
        beta = torch.stack(entries.beta)
        advantage = torch.stack(entries.advantage) # size([5, 8, 1])
        entropy = torch.stack(entries.entropy) # size([5, 8, 1])
        beta_advantage = torch.stack(entries.beta_advantage) # size([5, 8, 1])
        q = torch.stack(entries.q) # size([5, 8, 3])
        ret = torch.stack(entries.ret)
        option = torch.stack(entries.option)
        prev_option = torch.stack(entries.prev_option)
        init_state = torch.stack(entries.init_state)

        pp_list = []
        for log_p, a in zip(log_pi, action):
            pp = []
            for idx, p in enumerate(log_p):
                pp.append(log_p[idx][a[idx]])
            pp = torch.stack(pp)  # size([8, 1])
            pp_list.append(pp)
        pp_res = torch.stack(pp_list) # size([5, 8, 1]) size([rollout_length, num_envs, 1])

        pi_loss = -(pp_res * advantage.detach()) #- args.entropy_weight * entropy
        pi_loss = pi_loss.mean()

        q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()

        beta_loss = beta.gather(1, prev_option) * beta_advantage.detach() * (1 - init_state)
        beta_loss = beta_loss.mean()

        loss = pi_loss + q_loss + beta_loss

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), args.gradient_clip)
        self.optimizer.step()

        tb.logkv_mean('Loss', loss.item())


    def save(self, log_name, step):
        # memory_file_name = 'memory-' + str(step) + '.pkl'
        # memory_file = pjoin(log_name, memory_file_name)
        # pickle.dump(self.storage_save, open(memory_file, 'wb+'))

        model_file_name = 'model-' + str(step) + '.pt'
        model_file = pjoin(log_name, model_file_name)
        torch.save(self.network, model_file)


    def run_steps(self, args, eval_env, tb, logger):
        start = time.time()
        max_score = 0
        storage = Storage(args.rollout_length,
                          ['beta', 'option', 'beta_advantage', 'prev_option', 'init_state', 'eps'])
        if args.use_graph_info:
            obs, infos, graph_infos = self.envs.reset()
        else:  # pure
            obs, infos = self.envs.reset()

        if self.use_bidaf:
            obs_ids = encode_observation(obs, self.word2id)
            if self.state_all:
                look_ids = encode_observation([info['look'] for info in infos], self.word2id)
                inv_ids = encode_observation([info['inv'] for info in infos], self.word2id)

        elif not self.sent_transformer and not self.trans_network:   # pure
            states = self.build_state(obs, infos)
            states, _ = self.network.staternn(states)  # states: size([8, 384])
        # if args.use_graph_info:
        #     graph_state_reps = [g.graph_state_rep for g in graph_infos]
        #     g_t = self.state_gat.forward(graph_state_reps)  # size([8, 100])
        #     state_graph = torch.cat((states, g_t), dim=-1)  # size([8, 484])

        is_initial_states = tensor(np.ones((args.num_envs))).byte()
        prev_options = is_initial_states.clone().long()

        for step in range(1, args.max_steps + 1):
            tb.logkv('Step', step)

            acts = [info['valid'] for info in infos]
            act_sizes = [len(a) for a in acts]
            look = [info['look'] for info in infos]
            inv = [info['inv'] for info in infos]
            valid_ids = [self.encode(info['valid']) for info in infos]
            admissible_cmd_ids = list(itertools.chain.from_iterable(valid_ids))  # len: 32

            if self.use_bidaf:  # use_bidaf or not both will go this
                act_ids = encode_action([i['valid'] for i in infos], self.word2id)
                act_sizes = [len(a) for a in act_ids]
                if self.word_transformer:  # pass into original words
                    prediction = self.network((obs, look, inv), acts, act_sizes)  # q, log_pi, beta, pi

                elif self.state_all:  # state: (obs_ids, inv_ids, look_ids)
                    prediction = self.network((obs_ids, inv_ids, look_ids), act_ids, act_sizes)  # q, log_pi, beta, pi
                else:  # just obs
                    prediction = self.network(obs_ids, act_ids, act_sizes) # q, log_pi, beta, pi

            elif self.use_graph_info:
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                g_t = self.state_gat.forward(graph_state_reps)  # size([8, 100])
                state_graph = torch.cat((states, g_t), dim=-1)  # size([8, 484])
                prediction = self.network(state_graph, act_out, act_sizes)  # q, log_pi, beta, pi

            elif self.sent_transformer:  # without bidaf and graph
                prediction = self.network((obs, look, inv), admissible_cmd_ids, act_sizes)

            elif self.trans_network:
                states = self.build_state(obs, infos)
                prediction = self.network(states, admissible_cmd_ids, act_sizes)

            else:  # pure
                act_out = self.network.staternn.forward_enc_act(admissible_cmd_ids)  # size([32, 128])
                prediction = self.network(states, act_out, act_sizes)  # q, log_pi, beta, pi

            epsilon = args.random_option_prob(args.num_envs)
            options = self.sample_option(prediction, epsilon, prev_options, is_initial_states)

            pi = [p[o] for p, o in zip(prediction['pi'], options)]
            log_pi = [p[o] for p, o in zip(prediction['log_pi'], options)]

            actions, entropy = [], []
            for p in pi:
                dist = torch.distributions.Categorical(probs=p)
                actions.append(dist.sample())
                entropy.append(dist.entropy())
            actions = torch.stack(actions)
            entropy = torch.stack(entropy)

            action_strs = [info['valid'][idx] for info, idx in zip(infos, actions)]
            if self.use_graph_info:
                obs, rewards, dones, infos, graph_infos = self.envs.step(action_strs)
            else:
                obs, rewards, dones, infos = self.envs.step(action_strs)

            for done, info in zip(dones, infos):
                if info['score'] > max_score:
                    max_score = info['score']
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])

            if self.noisy_reward:
                noisy_rewards = []
                for r in rewards:
                    n = np.random.random()
                    if r < self.eps:
                        if n < 0.1:
                            r += self.noisy_value
                    else:
                        if n < 0.3:
                            r -= self.noisy_value
                    noisy_rewards.append(r)
                rewards = noisy_rewards

            storage.feed({'step': step,
                          'pi': pi,
                          'log_pi': log_pi,
                          'q': prediction['q'],
                          'beta': prediction['beta'],
                          'reward': tensor(rewards).unsqueeze(-1),
                          'option': options.unsqueeze(-1),
                          'prev_option': self.prev_options.unsqueeze(-1),
                          'entropy': entropy.unsqueeze(-1),
                          'action': actions.unsqueeze(-1),
                          'init_state': self.is_initial_states.unsqueeze(-1),
                          'eps': epsilon,
                          'mask': tensor(1 - dones).unsqueeze(-1),
                          'action_strs': action_strs})

            tb.logkv('Option', options.cpu().data.numpy())
            tb.logkv('Previous option', prev_options.cpu().data.numpy())
            tb.logkv('Reward', np.array(rewards))

            if self.use_bidaf:
                next_obs_ids = encode_observation(obs, self.word2id)
            # elif self.sent_transformer:
            #     acts = [info['valid'] for info in infos]
            #     act_sizes = [len(a) for a in acts]
            #     look = [info['look'] for info in infos]
            #     inv = [info['inv'] for info in infos]
            elif not self.sent_transformer and not self.trans_network:
                next_states = self.build_state(obs, infos)
                next_states, _ = self.network.staternn(next_states)

            is_initial_states = tensor(dones).byte()
            prev_options = options

            # if self.use_graph_info:
            #     graph_state_reps = [g.graph_state_rep for g in graph_infos]
            #     next_g_t = self.state_gat.forward(graph_state_reps)  # size([8, 100])
            #     next_state_graph = torch.cat((next_states, next_g_t), dim=-1)  # size([8, 484])
            #     state_graph = next_state_graph
            if step // args.num_envs % args.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            if len(storage.reward) >= args.rollout_length:
                tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
                # compute returns and advantage
                if self.use_bidaf:
                    act_ids = encode_action([i['valid'] for i in infos], self.word2id)
                    act_sizes = [len(a) for a in act_ids]
                elif not self.sent_transformer and not self.trans_network:
                    valid_ids = [self.encode(info['valid']) for info in infos]
                    act_sizes = [len(a) for a in valid_ids]
                    admissible_cmd_ids = list(itertools.chain.from_iterable(valid_ids))  # len: 32
                    act_out = self.network.staternn.forward_enc_act(admissible_cmd_ids)  # size([32, 128])

                with torch.no_grad():
                    if self.use_graph_info:
                        next_graph_state_reps = [g.graph_state_rep for g in graph_infos]
                        # next_g_t = self.target_state_gat.forward(next_graph_state_reps)
                        next_g_t = self.state_gat.forward(next_graph_state_reps)
                        next_state_graph = torch.cat((next_states, next_g_t), dim=-1)
                        prediction = self.target_network(next_state_graph, act_out, act_sizes)  # q, beta

                    elif self.use_bidaf:
                        obs_ids = next_obs_ids
                        look_ids = encode_observation([info['look'] for info in infos], self.word2id)
                        inv_ids = encode_observation([info['inv'] for info in infos], self.word2id)

                        if self.state_all:
                            prediction = self.target_network((obs_ids, inv_ids, look_ids), act_ids, act_sizes)  # q, log_pi, beta, pi
                        else:
                            prediction = self.target_network(obs_ids, act_ids, act_sizes)  # q, log_pi, beta, pi
                    elif self.sent_transformer:
                        # next state for target network
                        acts = [info['valid'] for info in infos]
                        act_sizes = [len(a) for a in acts]
                        look = [info['look'] for info in infos]
                        inv = [info['inv'] for info in infos]

                        prediction = self.target_network((obs, look, inv), acts, act_sizes)
                    elif self.trans_network:
                        next_states = self.build_state(obs, infos)
                        valid_ids = [self.encode(info['valid']) for info in infos]
                        act_sizes = [len(a) for a in valid_ids]
                        admissible_cmd_ids = list(itertools.chain.from_iterable(valid_ids))  # len: 32
                        prediction = self.target_network(next_states, admissible_cmd_ids, act_sizes)  # q, beta

                    else:  # pure
                        prediction = self.target_network(states, act_out, act_sizes)  # q, beta
                    storage.placeholder()

                    betas = prediction['beta'][self.worker_index, prev_options]  # size([8, 3]) -> size([8, 1])
                    q = prediction['q'][self.worker_index, prev_options]  # size([8, 3]) -> size([8, 1])

                    ret = (1 - betas) * q + betas * torch.max(prediction['q'], dim=-1)[0]
                    ret = ret.unsqueeze(-1)  # size([8, 1])

                for i in reversed(range(len(storage.reward))):
                    ret = storage.reward[i] + args.discount * storage.mask[i] * ret
                    adv = ret - storage.q[i].gather(1, storage.option[i])
                    storage.ret[i] = ret
                    storage.advantage[i] = adv

                    v = (1 - storage.eps[i]) * storage.q[i].max(dim=-1, keepdim=True)[0] + \
                        storage.eps[i] * storage.q[i].mean(-1).unsqueeze(-1)
                    q = storage.q[i].gather(1, storage.prev_option[i])
                    storage.beta_advantage[i] = q - v + args.termination_regularizer

                log_pi = storage.log_pi
                action = storage.action
                beta = torch.stack(storage.beta)
                advantage = torch.stack(storage.advantage)  # size([5, 8, 1])
                entropy = torch.stack(storage.entropy)  # size([5, 8, 1])
                beta_advantage = torch.stack(storage.beta_advantage)  # size([5, 8, 1])
                q = torch.stack(storage.q)  # size([5, 8, 3])
                ret = torch.stack(storage.ret)
                option = torch.stack(storage.option)
                prev_option = torch.stack(storage.prev_option)
                init_state = torch.stack(storage.init_state)

                pp_list = []
                for log_p, a in zip(log_pi, action):
                    pp = []
                    for idx, p in enumerate(log_p):
                        pp.append(log_p[idx][a[idx]])
                    pp = torch.stack(pp)  # size([8, 1])
                    pp_list.append(pp)
                pp_res = torch.stack(pp_list)  # size([5, 8, 1]) size([rollout_length, num_envs, 1])

                pi_loss = -(pp_res * advantage.detach()) - args.entropy_weight * entropy
                pi_loss = pi_loss.mean()

                q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()

                beta_loss = beta.gather(1, prev_option) * beta_advantage.detach() * (1 - init_state)
                beta_loss = beta_loss.mean()

                loss = pi_loss + q_loss + beta_loss

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.network.parameters(), args.gradient_clip)
                self.optimizer.step()
                storage.reset()

                tb.logkv_mean('Loss', loss.item())
                tb.logkv('Step', step)
                tb.logkv("Max score seen", max_score)
                tb.dumpkvs()

            if step % args.checkpoint_freq == 0:
                self.save(args.log_name, step)