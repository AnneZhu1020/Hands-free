import subprocess
import time
import os
import torch
import logger
import argparse
import jericho
from jericho.util import clean

from oc_agent import *
from vec_env import VecEnv
from env import JerichoEnv
from replay import *

torch.autograd.set_detect_anomaly(True)


class Drrn_option():
    def __init__(self, args):
        # assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
        self.options_tensor = torch.Tensor(np.arange(args.option_cnt))

    def configure_logger(self, log_dir, wandb):
        logger.configure(log_dir, format_strs=['log'])
        global tb
        tb = logger.Logger(log_dir, [logger.make_output_format('log', log_dir),
                                     logger.make_output_format('csv', log_dir),
                                     logger.make_output_format('stdout', log_dir)])
                                    #  logger.make_output_format('wandb', log_dir)])
        global log
        log = logger.log
        return tb, log

    def evaluate(self, agent, env, nb_episodes=1):
        with torch.no_grad():
            total_score = 0
            for ep in range(nb_episodes):
                log("Starting evaluation episode {}".format(ep))
                score = self.evaluate_episode(agent, env)
                log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
                total_score += score
            avg_score = total_score / nb_episodes
            return avg_score

    def evaluate_episode(self, agent, env):
        step = 0
        done = False
        ob, info = env.reset()
        state = agent.build_state([ob], [info])
        state, _ = agent.staternn(state)
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        while not done:
            valid_acts = info['valid']
            valid_ids = [agent.encode(valid_acts)]
            act_sizes = [len(a) for a in valid_ids]
            admissible_cmd_ids = list(itertools.chain.from_iterable(valid_ids))  # len: 32
            act_out = agent.staternn.forward_enc_act(admissible_cmd_ids)  # size([32, 128])

            prediction = agent.network(state, act_out, act_sizes)
            # choose argmax option
            q_option = prediction['q']
            option = q_option.argmax(dim=-1, keepdim=True)

            pi = prediction['pi'][0][option]
            # choose argmax pi
            action_values, action_idx = torch.max(pi, dim=-1)

            action_str = valid_acts[action_idx]
            log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values.item()))
            s = ''
            for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True),
                                             1):
                s += "{}){:.2f} {} ".format(idx, val.item(), act)
            log('Q-Values: {}'.format(s))
            ob, rew, done, info = env.step(action_str)
            log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
            step += 1
            log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
            state = agent.build_state([ob], [info])
            state, _ = agent.staternn(state)

        return info['score']

    def train(self, agent, eval_env, envs, args):
        start = time.time()
        obs, infos = envs.reset()
        states = agent.build_state(obs, infos)  # observation, description, inventory
        # states = agent.state2rnn(states)
        states, state_h = agent.staternn(states)  # size([8, 384])
        # states = tensor(states)

        valid_ids = [agent.encode(info['valid']) for info in infos]  # ['open mailbox', 'north', 'south', 'west']

        option_used = ''
        current_vals = agent.get_option_vals(states)  # size([8, 3]) size([8, 384])
        option, options_prob = softmax(current_vals)  # size([8, 1]) size([8, 3])

        option_switches = 1

        for step in range(1, args.max_steps + 1):
            extended_states = []
            for state, option_idx in zip(states, option):
                extended_states_tmp = torch.cat((state, agent.embeddings[option_idx].squeeze(0)), -1)  # size([8, 387])
                extended_states.append(extended_states_tmp)
            extended_states = torch.stack(extended_states)  # size([8,387])

            # 1. actor action based on option
            if args.action_discrete:
                act_index, act_value, act_values = agent.act_option(extended_states, valid_ids)
                act_ids = [valid_ids[batch][idx] for batch, idx in enumerate(act_index)]
                act_value = torch.stack(act_value)

            else:
                action = agent.act_option(extended_states, valid_ids)
            # extended_states_clone = extended_states.clone()

            # 2. interact with env
            action_strs = [info['valid'][idx] for info, idx in zip(infos, act_index)]
            obs, rewards, dones, infos = envs.step(action_strs)

            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])

            next_states_tmp = agent.build_state(obs, infos)
            # next_states = agent.state2rnn(next_states)
            next_states, next_states_h = agent.staternn(next_states_tmp)  # size([8, 384])

            # extended_next_states = []
            # for state, option_idx in zip(next_states, option):
            #     extended_next_states_tmp = torch.cat((state, agent.embeddings[option_idx].squeeze(0)), -1)  # size([8, 387])
            #     extended_next_states.append(extended_next_states_tmp)
            # extended_next_states = torch.stack(extended_next_states)  # size([8,387])

            extended_next_states = []
            for state in next_states:
                extended_next_states_tmp = torch.cat((state.repeat(args.option_cnt, 1),
                                                      agent.encode_option(self.options_tensor)), -1)  # size([3, 387])
                extended_next_states.append(extended_next_states_tmp)
            extended_next_states = torch.stack(extended_next_states)  # size([8, 3, 387])
            b = torch.arange(args.num_envs)
            extended_next_state = extended_next_states[b, option.squeeze(-1), :]  # size([8, 387])

            # 3. append into memory
            next_valids = [agent.encode(info['valid']) for info in infos]
            # for state, ex_state, act, act_val, act_vals, rew, next_state, ex_next_state, ex_next_states, next_valid_acts, done in \
            #     zip(states.detach(), extended_states.detach().cpu().numpy(), act_ids, act_value, act_values, rewards, next_states.detach(),
            #         extended_next_state_clone.detach(), extended_next_states_clone.detach(), next_valids, dones):
            #     agent.observe(state, ex_state, act, act_val, act_vals, rew, next_state, ex_next_state, ex_next_states, next_valid_acts, done)
            for state, ex_state, act, act_val, act_vals, rew, next_state, ex_next_state, ex_next_states, next_valid_acts, done in \
                    zip(states.detach().cpu().numpy().tolist(), extended_states.detach().cpu().numpy().tolist(),
                        act_ids, act_value.detach().cpu().numpy().tolist(), act_values, rewards, next_states,
                        extended_next_state.detach().cpu().numpy().tolist(),
                        extended_next_states.detach().cpu().numpy().tolist(),
                        next_valids, dones):
                agent.observe(state, ex_state, act, act_val, act_vals, rew, next_state, ex_next_state, ex_next_states,
                              next_valid_acts, done)

            # 3.1 reset hidden
            done_mask_tt = (torch.tensor(dones)).float().cuda().unsqueeze(1)

            # 4. get beta based on next_state
            # extended_next_state = tensor(extended_next_state)
            beta_next = agent.beta(extended_next_state)  # size([8, 387])
            beta_next_np = beta_next.detach().cpu().numpy()  # size([8, 1])
            new_option = option
            option_termination = sample_sigmoid(beta_next_np)

            new_option_list, new_options_prob_list = [], []

            next_vals = agent.get_option_vals(next_states)  # size([8, 3])
            for idx in range(len(option_termination)):
                # for option_t, next_state in zip(option_termination, next_states):
                if option_termination[idx]:
                    new_o, options_p = softmax(next_vals[idx])
                    if new_o != option[idx]:
                        option_switches += 1
                    new_option_list.append(new_o.squeeze(0))
                    new_options_prob_list.append(options_p.squeeze(0))
                else:
                    new_option_list.append(option[idx].squeeze(0))
                    new_options_prob_list.append(options_prob[idx].squeeze(0))
            new_option = torch.stack(new_option_list)  # size([8, 1])
            new_options_prob = torch.stack(new_options_prob_list)  # size([8, 3])

            old_option = option
            option = new_option.clone().detach()

            states = next_states.clone().detach()
            valid_ids = next_valids

            agent.staternn.reset_hidden(done_mask_tt)

            if step % args.log_freq == 0:
                tb.logkv('Step', step)
                tb.logkv("FPS", int((step * envs.num_envs) / (time.time() - start)))
                tb.dumpkvs()
            if step % args.update_freq == 0:
                agent.train_model()
                # loss = agent.update()
                # if loss is not None:
                #     tb.logkv_mean('Loss', loss)
            if step % args.checkpoint_freq == 0:
                agent.save(args.log_name)
            if step % args.eval_freq == 0:
                eval_score = self.evaluate(agent, eval_env)
                tb.logkv('EvalScore', eval_score)
                tb.dumpkvs()

    def start_redis(self):
        print('Starting Redis')
        subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no', '--port 6383'])
        time.sleep(1)

    def start_openie(self, install_path):
        print('Starting OpenIE from', install_path)
        subprocess.Popen(['java', '-mx8g', '-cp', '*', \
                          'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', \
                          '-port', '9040', '-timeout', '15000'], cwd=install_path)  # , '-quiet'
        time.sleep(1)

    def interactive_run(self, env):
        ob, info = env.reset()
        while True:
            print(clean(ob), 'Reward', reward, 'Done', done, 'Valid', info)
            ob, reward, done, info = env.step(input())


    def run_steps(self, args, agent, eval_env):
        start = time.time()
        while True:
            # if args.save_interval and not agent.total_steps % args.save_interval:
            #     agent.save('data/%d' % agent.total_steps)
            # if args.log_interval and not agent.total_steps % args.log_interval:
            #     agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, args.log_interval / (time.time() - t0)))
            #     t0 = time.time()
            # if args.eval_interval and not agent.total_steps % args.eval_interval:
                # agent.eval_episodes()
            if args.max_steps and agent.total_steps >= args.max_steps:
                # agent.close()
                break

            if agent.total_steps % args.log_freq == 0:
                tb.logkv('Step', agent.total_steps)
                tb.logkv("FPS", int((agent.total_steps * envs.num_envs) / (time.time() - start)))
                tb.dumpkvs()
            if agent.total_steps % args.checkpoint_freq == 0:
                agent.save()
            if agent.total_steps % args.eval_freq == 0:
                eval_score = self.evaluate(agent, eval_env)
                tb.logkv('EvalScore', eval_score)
                tb.dumpkvs()

            agent.step(tb)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs_Apr')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')

    parser.add_argument('--rom_path', default='../roms/zork1.z5')
    parser.add_argument('--tsv_file', default='../data/zork1_entity2id.tsv')

    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=5000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=8, type=int)
    parser.add_argument('--memory_size', default=500000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--gamma', default=.5, type=float)  # TODO: try with 0.5 in kga2c
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)

    # option configuration
    parser.add_argument('--option_cnt', help='option count', type=int, default=3, choices=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument('--action_discrete', help='whether the action space is discrete or deterministic'
                                                  'if deterministic, use gaussian policy; else, dqn', default=True,
                        type=bool)
    parser.add_argument('--update_step', default=2048, type=int)
    parser.add_argument('--entropy_weight', default=0.01, type=float)
    parser.add_argument('--gradient_clip', default=5, type=float)
    parser.add_argument('--rollout_length', default=8, type=int)
    parser.add_argument('--target_network_update_freq', default=1000, choices=[1000, 200])
    parser.add_argument('--learning_rate', default=0.0001, type=float, choices=[0.0001, 0.0003, 0.001])
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--termination_regularizer', default=0.01, type=float)

    parser.add_argument('--use_graph_info', default=False, type=bool)
    parser.add_argument('--openie_path', default='../stanford-corenlp-full-2018-10-05')
    parser.add_argument('--masking', default='kg', choices=['kg', 'interactive', 'none'], help='Type of object masking applied')
    parser.add_argument('--gat_emb_size', default=50, type=int)
    parser.add_argument('--statenetwork_emb_size', default=50, type=int)
    parser.add_argument('--dropout_ratio', default=0.2, type=float)
    parser.add_argument('--max_stuck_steps', default=100, choices=[100, 50, 10], type=int)

    parser.add_argument('--use_bidaf', default=True, type=bool)
    parser.add_argument('--bidaf_hidden_dim', default=32, type=int)
    parser.add_argument('--keep_prob', default=1, type=float, help="0 <= keep_prob <= 1", choices=[1, 0.9])
    parser.add_argument('--glove_file', default='./glove/dict.pt', type=str)
    parser.add_argument('--glove_dim', default=100, type=int)
    parser.add_argument('--norm', default='layer', help='layer/batch')
    parser.add_argument('--max_obs_seq_len', default=400, type=int)
    parser.add_argument('--max_act_seq_len', default=10, type=int)
    parser.add_argument('--act_num_limit', default=15, type=int)
    parser.add_argument('--act_len_limit', default=10, type=int)

    parser.add_argument('--adv_variant', default=False, type=bool)
    parser.add_argument('--single_att2cont', default=False, type=bool)
    parser.add_argument('--single_att2que', default=False, type=bool)

    parser.add_argument('--state_all', default=False, type=bool)

    parser.add_argument('--noisy_reward', default=False, type=bool)
    parser.add_argument('--noisy_value', default=5.0, type=float)

    parser.add_argument('--sent_transformer', default=True, type=bool)
    parser.add_argument('--word_transformer', default=False, type=bool)
    parser.add_argument('--trans_network', default=False, type=bool)

    parser.add_argument('--wandb', default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    print(args)

    Drrn_option = Drrn_option(args)

    game = args.rom_path.split('/')[-1].split('.')[0]
    tmp = '-num_env' + str(args.num_envs) + '-rollout_length' + str(args.rollout_length) + '-optioncnt' + str(args.option_cnt) \
          + '-discount' + str(args.discount) + '-clip' + str(args.gradient_clip) \
          + '-adam-lr' + str(args.learning_rate) + '-targetupdatefreq' + str(args.target_network_update_freq) \
          + '-entropy_w' + str(args.entropy_weight) + '-hidden' + str(args.bidaf_hidden_dim)
    log_name = args.output_dir + '-' + game + '/'

    if args.noisy_reward:
        log_name += 'noisyReward' + str(args.noisy_value)
    if args.sent_transformer:
        log_name += '2_sent_trans'
    if args.word_transformer:
        log_name += 'word_trans'
    if args.trans_network:
        log_name += '2_trans_network-emb64-layer1-head2'

    if args.single_att2cont:
        log_name += '-att2cont'
    if args.single_att2que:
        log_name += '-att2que'
    if args.use_graph_info:
        log_name += '-graph-maxstuck' + str(args.max_stuck_steps)
    if args.adv_variant:
        log_name += '-adv_var'
    if args.state_all:
        log_name += '-allstate-target_update'
    if args.use_bidaf:
        log_name += '-dropout_prob' + str(args.keep_prob)

    log_name += tmp
    print(log_name)
    args.log_name = log_name
    tb, logger = Drrn_option.configure_logger(log_name, args.wandb)

    Drrn_option.start_redis()
    if args.use_graph_info:
        Drrn_option.start_openie(args.openie_path)
    # rom_path, seed, spm_model, tsv_file, step_limit = None, stuck_steps = 10, gat = True
    env = JerichoEnv(args.rom_path, args.seed, args.spm_path, args.tsv_file, args.env_step_limit, args.use_graph_info,
                     args.max_stuck_steps)
    envs = VecEnv(args.num_envs, env)
    env.create()  # Create the environment for evaluation
    if args.use_graph_info:
        agent = OptionCriticAgent(args, envs, env.vocab)
    else:
        agent = OptionCriticAgent(args, envs)

    agent.run_steps(args, env, tb, logger)

    # Drrn_option.train(agent, env, envs, args)
    del Drrn_option
