import torch
import numpy as np
from pathlib import Path
import spacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


def wrap_observation_tensor(state_ids, device, maxlen):
    state_ids, state_lengths = wrap_pad_sequences(sequences=state_ids, maxlen=maxlen)

    batch_state = (torch.from_numpy(state_ids)
                   .to(device=device, dtype=torch.long))

    batch_state_length = (torch.from_numpy(np.asarray(state_lengths))
                          .to(device=device, dtype=torch.long))

    batch_state_length.clamp_max_(max=batch_state.size(1))

    return batch_state, batch_state_length


def wrap_pad_sequences(sequences, maxlen, dtype='int32', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    maxlen = min(np.max(lengths), maxlen)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[:maxlen]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        # post padding
        x[idx, :len(trunc)] = trunc
    return x, lengths

def encode_observation(observation, word2id, ignore_action=False):
    nlp_pipe = spacy.load('en_core_web_sm')

    remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS',
              'UNK', 'unk', 'sos', '<', '>']
    obs_list = []
    for obs in observation:
        for rm in remove:
            obs = obs.replace(rm, '')
        obs = obs.replace('.\n', '.')
        obs = obs.replace('. \n', '.')
        obs = obs.replace('\n', '.')
        obs = obs.split('|')

        obs_list.append(obs)


    # observation = observation.replace('.\n', '.')
    # observation = observation.replace('. \n', '.')
    # observation = observation.replace('\n', '.')
    # observation = observation.split('|')
    # if ignore_action:
    #     observation = observation[:-1]

    # flatten_tokens = ['<ACT>']
    # spacy_ret = [self.word2id['<ACT>']]
    res = []
    for obs in obs_list:
        spacy_ret = []
        doc = nlp_pipe(obs[0])
        spacy_ret.append(word2id['<ACT>'])
        text_tokens = [token.text.lower()
                       for token in doc if token.text.lower() != ' ']
        text_tokens = [token
                       if token != '..' else '.'
                       for token in text_tokens]
        text_tokens = [token
                       if token in word2id else '<OOV>'
                       for token in text_tokens]
        spacy_ret.extend([word2id[token] for token in text_tokens])
        spacy_ret.append(word2id['<ACT>'])
        res.append(spacy_ret)
    return res


def encode_action(action_list_ids, word2id):
    nlp_pipe = spacy.load('en_core_web_sm')
    res_list = []
    for action_list in action_list_ids:
        res = []
        for action in action_list:
            spacy_ret = []
            doc = nlp_pipe(action)
            # spacy_ret.append(word2id['<ACT>'])
            text_tokens = [token.text.lower()
                           for token in doc if token.text.lower() != ' ']
            text_tokens = [token
                           if token != '..' else '.'
                           for token in text_tokens]
            text_tokens = [token
                           if token in word2id else '<OOV>'
                           for token in text_tokens]
            spacy_ret.extend([word2id[token] for token in text_tokens])
            # spacy_ret.append(word2id['<ACT>'])
            res.append(spacy_ret)
        res_list.append(res)
    return res_list


def wrap_action_tensor(act_ids, device, act_num_limit, act_len_limit):
    action_ids = pad_group_sequences(sequences=act_ids, maxgroup=act_num_limit, maxlen=act_len_limit)
    batch_template = torch.from_numpy(action_ids).to(device=device, dtype=torch.long)

    return batch_template

# B x vK x vL --> B x K x L
def pad_group_sequences(sequences, maxgroup, maxlen, dtype='int32', value=0.):
    group_sizes = [len(g) for g in sequences]
    lengths = [len(s) for g in sequences for s in g]
    nb_samples = len(sequences)
    maxlen = min(np.max(lengths), maxlen)

    maxgroup = min(np.max(group_sizes), maxgroup)

    x = (np.ones((nb_samples, maxgroup, maxlen)) * value).astype(dtype)
    for idx, t in enumerate(sequences):
        if len(t) == 0:
            continue  # empty list was found
        for g_idx, g in enumerate(t):
            if len(g) == 0:
                continue
            if g_idx >= maxgroup:
                break
            trunc = g[:maxlen]
            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            # post padding
            x[idx, g_idx, :len(trunc)] = trunc
    return x