from os.path import basename
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import *
from jericho.defines import *
import redis

from representations import StateAction

GraphInfo = collections.namedtuple('GraphInfo',
                                   'objs, ob_rep, act_rep, graph_state, graph_state_rep, admissible_actions, admissible_actions_rep')

import sentencepiece as spm

MAX_STUCK_STEPS = 100

def load_vocab_rev(env):
    vocab = {i + 2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: idx for idx, v in vocab.items()}
    return vocab_rev


def load_vocab_withgraph(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev


def clean_obs(s):
    garbage_chars = ['*', '-', '!', '[', ']']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()


class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, spm_path, tsv_file,
                 step_limit=None, use_graph_info=False, max_stuck_steps=100, gat=True):
        self.rom_path = rom_path
        self.bindings = load_bindings(rom_path)
        self.act_gen = TemplateActionGenerator(self.bindings)
        self.max_word_len = self.bindings['max_word_length']
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.env = None
        self.conn = None

        self.vocab_rev = None
        self.use_graph_info = use_graph_info


        if use_graph_info:
            self.spm_model = spm.SentencePieceProcessor()
            self.spm_model.Load(spm_path)
            self.tsv_file = tsv_file
            self.step_limit = step_limit
            self.gat = gat
            self.conn_valid = None
            self.conn_openie = None
            self.vocab = None

            self.state_rep = None
            self.stuck_steps = 0
            self.max_stuck_steps = max_stuck_steps
            self.valid_steps = 0

    def create(self):
        self.env = FrotzEnv(self.rom_path, self.seed)
        if self.use_graph_info:
            self.vocab, self.vocab_rev = load_vocab_withgraph(self.env)
            self.conn_valid = redis.Redis(host='localhost', port=6383, db=0)
            self.conn_openie = redis.Redis(host='localhost', port=6383, db=1)
        else:
            self.vocab_rev = load_vocab_rev(self.env)
            self.conn = redis.Redis(host='localhost', port=6383, db=0)
            self.conn.flushdb()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        if not done:  # done -> False
            try:
                save = self.env.save_str()
                look, _, _, _ = self.env.step('look')
                info['look'] = look
                self.env.load_str(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv
                self.env.load_str(save)
                # Get the valid actions for this state
                world_state_hash = self.env.get_world_state_hash()
                if self.use_graph_info:
                    valid = self.conn_valid.get(world_state_hash)
                else:
                    valid = self.conn.get(world_state_hash)
                if valid is None:
                    objs = [o[0] for o in self.env.identify_interactive_objects(ob)]
                    obj_ids = [self.vocab_rev[o[:self.bindings['max_word_length']]] for o in objs]
                    acts = self.act_gen.generate_template_actions(objs, obj_ids)
                    valid = self.env.find_valid_actions(acts)
                    redis_valid_value = '/'.join([str(a) for a in valid])
                    if self.use_graph_info:
                        self.conn_valid.set(world_state_hash, redis_valid_value)
                    else:
                        self.conn.set(world_state_hash, redis_valid_value)
                    valid = [a.action for a in valid]
                else:
                    valid = valid.decode('cp1252')
                    if valid:
                        valid = [eval(a).action for a in valid.split('/')]
                    else:
                        valid = []
                if len(valid) == 0:
                    valid = ['wait', 'yes', 'no']
                info['valid'] = valid
            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean(ob), done, info))

        if self.use_graph_info:
            if self.env.world_changed() or done:
                self.valid_steps += 1
                self.stuck_steps = 0
            else:
                self.stuck_steps += 1

        self.steps += 1

        if self.use_graph_info:
            if self.step_limit and self.valid_steps >= self.step_limit \
                    or self.stuck_steps > self.max_stuck_steps:
                done = True
        else:
            if self.step_limit and self.steps >= self.step_limit:
                done = True

        if self.use_graph_info:
            if done:
                graph_info = GraphInfo(objs=['all'],
                                       ob_rep=self.state_rep.get_obs_rep(ob, ob, ob, action),
                                       act_rep=self.state_rep.get_action_rep_drqa(action),
                                       graph_state=self.state_rep.graph_state,
                                       graph_state_rep=self.state_rep.graph_state_rep,
                                       admissible_actions=[],
                                       admissible_actions_rep=[])
            else:
                graph_info = self._build_graph_rep(action, ob)
            return ob, reward, done, info, graph_info

        else:
            return ob, reward, done, info

    # def reset(self):
    #     initial_ob, info = self.env.reset()
    #     save = self.env.save_str()
    #     look, _, _, _ = self.env.step('look')
    #     info['look'] = look
    #     self.env.load_str(save)
    #     inv, _, _, _ = self.env.step('inventory')
    #     info['inv'] = inv
    #     self.env.load_str(save)
    #     objs = [o[0] for o in self.env.identify_interactive_objects(initial_ob)]
    #     acts = self.act_gen.generate_actions(objs)
    #     valid = self.env.find_valid_actions(acts)
    #     info['valid'] = valid
    #     self.steps = 0
    #     return initial_ob, info

    def _get_admissible_actions(self, objs):
        ''' Queries Redis for a list of admissible actions from the current state. '''
        obj_ids = [self.vocab_rev[o[:self.max_word_len]] for o in objs]
        world_state_hash = self.env.get_world_state_hash()
        admissible = self.conn_valid.get(world_state_hash)
        if admissible is None:
            possible_acts = self.act_gen.generate_template_actions(objs, obj_ids)
            admissible = self.env.find_valid_actions(possible_acts)
            redis_valid_value = '/'.join([str(a) for a in admissible])
            self.conn_valid.set(world_state_hash, redis_valid_value)
        else:
            try:
                admissible = [eval(a.strip()) for a in admissible.decode('cp1252').split('/')]
            except Exception as e:
                print("Exception: {}. Admissible: {}".format(e, admissible))
        return admissible

    def _build_graph_rep(self, action, ob_r):
        ''' Returns various graph-based representations of the current state. '''
        objs = [o[0] for o in self.env.identify_interactive_objects(ob_r)]
        objs.append('all')
        admissible_actions = self._get_admissible_actions(objs)
        admissible_actions_rep = [self.state_rep.get_action_rep_drqa(a.action) \
                                  for a in admissible_actions] \
                                      if admissible_actions else [[0] * 20]
        try: # Gather additional information about the new state
            save_str = self.env.save_str()
            ob_l = self.env.step('look')[0]
            self.env.load_str(save_str)
            ob_i = self.env.step('inventory')[0]
            self.env.load_str(save_str)
        except RuntimeError:
            # print('RuntimeError: {}, Done: {}, Info: {}'.format(clean_obs(ob_r), done, info))
            print('RuntimeError: {}'.format(clean_obs(ob_r)))

            ob_l = ob_i = ''
        ob_rep = self.state_rep.get_obs_rep(ob_l, ob_i, ob_r, action)
        cleaned_obs = clean_obs(ob_l + ' ' + ob_r)
        openie_cache = self.conn_openie.get(cleaned_obs)
        if openie_cache is None:
            rules, tocache = self.state_rep.step(cleaned_obs, ob_i, objs, action, cache=None, gat=self.gat)
            self.conn_openie.set(cleaned_obs, str(tocache))
        else:
            openie_cache = eval(openie_cache.decode('cp1252'))
            rules, _ = self.state_rep.step(cleaned_obs, ob_i, objs, action, cache=openie_cache, gat=self.gat)
        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action)
        return GraphInfo(objs, ob_rep, action_rep, graph_state, graph_state_rep,\
                         admissible_actions, admissible_actions_rep)

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.save_str()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.load_str(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.load_str(save)
        objs = [o[0] for o in self.env.identify_interactive_objects(initial_ob)]
        acts = self.act_gen.generate_actions(objs)
        valid = self.env.find_valid_actions(acts)
        info['valid'] = valid
        self.steps = 0

        if self.use_graph_info:
            self.stuck_steps = 0
            self.valid_steps = 0
            self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
                                         self.tsv_file, self.max_word_len)
            graph_info = self._build_graph_rep('look', initial_ob)
            return initial_ob, info, graph_info

        else:
            return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def close(self):
        self.env.close()
