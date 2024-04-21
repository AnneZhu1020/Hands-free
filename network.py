import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
from torch.nn.utils import rnn
from collections import namedtuple
# from sentence_transformers import SentenceTransformer
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from fastNLP.embeddings import BertEmbedding
from fastNLP import Vocabulary
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
State = namedtuple('State', ('obs', 'description', 'inventory'))


class BaseNet:
    def __init__(self):
        pass

    def reset_noise(self):
        pass


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


# Adapted from https://github.com/saj1919/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, args, body, action_dim, word_mat=None):
        super(OptionCriticNet, self).__init__()
        self.body = body
        self.phi = None
        self.use_bidaf = args.use_bidaf
        self.state_all = args.state_all
        self.sent_transformer = args.sent_transformer
        if self.sent_transformer:
            in_feature = self.body.layers[0].in_features
            out_feature = self.body.layers[0].out_features
            self.emb_mlp_state = nn.Linear(in_feature * 3, in_feature)
            self.emb_mlp_action = nn.Linear(in_feature, out_feature)
        self.word_transformer = args.word_transformer
        self.trans_network = args.trans_network
        if self.trans_network:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(args.spm_path)
            self.packedSequence = PackedSequence(len(self.sp), args.embedding_dim)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
            self.gru_output_trans = nn.GRU(args.embedding_dim, args.hidden_dim)
            self.gru_packed_state = nn.GRU(args.embedding_dim, args.hidden_dim)
            self.fc_output = layer_init(nn.Linear(args.hidden_dim, args.option_cnt * action_dim))

        if args.use_bidaf:
            self.max_obs_seq_len = args.max_obs_seq_len
            self.max_act_seq_len = args.max_act_seq_len
            self.qc_att = BiAttention(args.bidaf_hidden_dim*2, 1 - args.keep_prob, args.single_att2cont, args.single_att2que)
            self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
            self.act_num_limit = args.act_num_limit
            self.act_len_limit = args.act_len_limit
            if args.word_transformer:
                self.word_dim = 768
            else:
                self.word_dim = args.glove_dim

            self.bidaf_rnn = EncoderRNN(self.word_dim, args.bidaf_hidden_dim, 1,
                                        concat=True, bidir=True, layernorm='None', return_last=True)
            if args.single_att2cont or args.single_att2que:
                self.fc_pi = layer_init(nn.Linear(args.bidaf_hidden_dim * 2, args.option_cnt * action_dim))
            else:
                self.fc_pi = layer_init(nn.Linear(args.bidaf_hidden_dim * 4, args.option_cnt * action_dim))

            self.fc_q = layer_init(nn.Linear(args.bidaf_hidden_dim * 2, args.option_cnt))
            self.fc_beta = layer_init(nn.Linear(args.bidaf_hidden_dim * 2, args.option_cnt))

        else:
            self.fc_pi = layer_init(nn.Linear(body.feature_dim + 128, args.option_cnt * action_dim))
            self.fc_q = layer_init(nn.Linear(body.feature_dim, args.option_cnt))
            self.fc_beta = layer_init(nn.Linear(body.feature_dim, args.option_cnt))

        self.to(device)

    def forward(self, state, act_out, act_sizes):
        if self.use_bidaf:
            # state_ids <- state: obs_ids
            # admissible_cmd_ids <- act_out

            if self.word_transformer:
                obs, inv, look = state

                vocab = Vocabulary()
                context_obs_output = []
                context_obs_lens = []
                for o in obs:
                    vocab.add_word_lst(o.split())
                    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', pool_method='max')
                    words = torch.LongTensor(
                        [[vocab.to_index(word) for word in o.split()]])
                    emb = embed(words).to(device)
                    context_obs_output.append(emb)  # [1, 50, 768]
                    context_obs_lens.append(emb.size(1))

                context_inv_output = []
                context_inv_lens = []
                for i in inv:
                    vocab.add_word_lst(i.split())
                    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', pool_method='max')
                    words = torch.LongTensor(
                        [[vocab.to_index(word) for word in i.split()]])
                    emb = embed(words).to(device)
                    context_inv_output.append(tensor(emb))
                    context_inv_lens.append(emb.size(1))

                context_look_output = []
                context_look_lens = []
                for l in look:
                    vocab.add_word_lst(l.split())
                    embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', pool_method='max')
                    words = torch.LongTensor(
                        [[vocab.to_index(word) for word in l.split()]])
                    emb = embed(words).to(device)
                    context_look_output.append(tensor(emb))
                    context_look_lens.append(emb.size(1))

                query_ids = wrap_action_tensor(act_out, device, self.act_num_limit, self.act_len_limit) # size([8, 4, 4])

                que_output = []
                que_num = []
                for act in act_out:
                    que_output_list = []
                    for a in act:
                        vocab.add_word_lst(a[0].split())
                        embed = BertEmbedding(vocab, model_dir_or_name='en-base-cased', layers='-1', pool_method='max')
                        words = torch.LongTensor(
                            [[vocab.to_index(word) for word in a[0].split()]])
                        emb = embed(words).to(device)
                        que_output_list.append(tensor(emb))
                    que_num.append(len(que_output_list))
                    que_output.append(que_output_list)

            else:  # glove embedding
                if self.state_all:
                    obs_ids, inv_ids, look_ids = state
                    context_obs_ids, context_obs_lens = wrap_observation_tensor(obs_ids, device,
                                                                        maxlen=self.max_obs_seq_len)  # size([8, 64])
                    context_inv_ids, context_inv_lens = wrap_observation_tensor(inv_ids, device,
                                                                        maxlen=self.max_obs_seq_len)
                    context_look_ids, context_look_lens = wrap_observation_tensor(look_ids, device,
                                                                        maxlen=self.max_obs_seq_len)
                    context_lens = context_obs_lens + context_inv_lens + context_look_lens
                else:
                    context_ids, context_lens = wrap_observation_tensor(state, device, maxlen=self.max_obs_seq_len) # size([8, 64])
                    cont_len = context_ids.size(1)

                query_ids = wrap_action_tensor(act_out, device, self.act_num_limit, self.act_len_limit) # size([8, 4, 4])

                que_num, que_len = query_ids.size(1), query_ids.size(2)
                bsz = query_ids.size(0)

                with torch.no_grad():
                    # (B x K) x Lq
                    que_mask = (query_ids > 0).float().view(-1, que_len)
                    if self.state_all:
                        context_obs_output = self.word_emb(context_obs_ids)  # size([8, context_lens, 100])
                        context_inv_output = self.word_emb(context_inv_ids)  # size([8, context_lens, 100])
                        context_look_output = self.word_emb(context_look_ids)  # size([8, context_lens, 100])
                    else:
                        context_output = self.word_emb(context_ids)  # size([8, context_lens, 100])
                    que_output = self.word_emb(query_ids)  # size([8, que_num, que_len, 100])

            # (B x K) x Lc x H
            if self.state_all:
                phi_obs_cont, phi = self.bidaf_rnn(torch.stack(context_obs_output).squeeze(1), tensor(context_obs_lens)) # size([8, cont_len, 64])
                phi_inv_cont, phi = self.bidaf_rnn(torch.stack(context_inv_output).squeeze(1), tensor(context_inv_lens)) # size([8, cont_len, 64])
                phi_look_cont, phi = self.bidaf_rnn(torch.stack(context_look_output).squeeze(1), tensor(context_look_lens)) # size([8, cont_len, 64])
                phi_cont = torch.cat((phi_obs_cont, phi_inv_cont, phi_look_cont), dim=1)  # size([8, 102, 64])
                cont_len = phi_cont.size(1)
            else:
                phi_cont, phi = self.bidaf_rnn(context_output, context_lens) # size([8, cont_len, 64])  size([8, 64, 64])

            context_output = phi_cont.unsqueeze(dim=1)\
                                .expand(-1, que_num, -1, -1)\
                                .contiguous()\
                                .view(bsz * que_num, cont_len, -1)  # size([bsz * que_num, cont_len, 64])
            # (B x K) x Lq x H
            que_output = (self.bidaf_rnn(que_output.view(-1, que_len, self.word_dim))[0]
                           .view(bsz * que_num, que_len, -1))  # size([bsz * que_num, que_len, 64])

            z = self.qc_att(context_output, que_output, que_mask)  # size([bsz * que_num, 128])
            z = self.fc_pi(z)  # size([bsz * que_num, option_cnt])

        elif self.trans_network:
            states = State(*zip(*state))
            packed_action = self.packedSequence(states.obs)  # size([83, 8, 128])
            packed_look = self.packedSequence(states.description)  # size([30, 8, 128])
            packed_inv = self.packedSequence(states.inventory)  # size([7, 8, 128])
            packed_state = torch.cat((packed_action, packed_look, packed_inv), dim=0)  # size([120, 8, 128])
            packed_state = torch.transpose(packed_state, 0, 1)  # size([8, 120, 128])

            phi, _ = self.gru_packed_state(packed_state)
            phi = phi[:,-1,:]  # size([8, 128])

            packed_state = torch.cat([packed_state[i].repeat(j, 1, 1) for i, j in enumerate(act_sizes)],
                                  dim=0)  # size([32, 120, 128])

            packed_action = self.packedSequence(act_out)  # size([2, 32, 128])
            packed_action = torch.transpose(packed_action, 0, 1)  # size([32, 2, 128])

            packed_state_action = torch.cat((packed_state, packed_action), dim=1)  # size([32, 122, 128])
            # into transformer encoder
            output_trans = self.transformer_encoder(packed_state_action)  # size([32, 122, 128])
            output, _ = self.gru_output_trans(output_trans)

            z = self.fc_output(output[:,-1,:])  # size([32, 5])

        else:
            if self.sent_transformer:
                # encode using sentence-transformers, then directly into Q-value estimation
                obs, look, inv = state
                model = SentenceTransformer('all-MiniLM-L6-v2')
                # state with words
                obs_emb = tensor(model.encode(obs))
                look_emb = tensor(model.encode(look))
                inv_emb = tensor(model.encode(inv))

                act_out = torch.cat([tensor(model.encode(a)) for a in act_out])
                state = torch.cat((obs_emb, look_emb, inv_emb), dim=1)
                state = self.emb_mlp_state(state)
                act_out = self.emb_mlp_action(act_out)

            phi = self.body(state)  # size([8, 128])
            state_out = torch.cat([phi[i].repeat(j, 1) for i, j in enumerate(act_sizes)],
                                  dim=0)  # size([32, 128])
            z = torch.cat((state_out, act_out), dim = 1) # Concat along hidden_dim size([32, 192])

            z = self.fc_pi(z)  # size([32, 3])
            # pi = pi.view(-1, self.num_options, self.action_dim)
            # z = z.view(-1, self.num_options, self.action_dim)

        if self.use_bidaf:
            pi = [z[i*que_num:i*que_num+j] for i,j in zip(torch.arange(bsz), act_sizes)]
        else:
            pi = z.split(act_sizes)

        log_pi = [F.log_softmax(p.transpose(0, 1)) for p in pi]  # 8 tuple: size([3, 4])
        pi = [F.softmax(p.transpose(0, 1)) for p in pi]
        # print("phi.size(): ", phi.size())  # size([3, 64])
        # print("self.fc_q: ", self.fc_q)
        q = self.fc_q(phi)  # size([8, option_cnt]) num_envs, option_cnt
        beta = F.sigmoid(self.fc_beta(phi))  # size([8, option_cnt]) num_envs, option_cnt
        return {'log_pi': log_pi,
                'pi': pi,
                'q': q,
                'beta': beta}


class PackedSequence(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(PackedSequence, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # num_embeddings, embedding_dim

    def forward(self, input):
        lengths = torch.tensor([len(n) for n in input], dtype=torch.long, device=device)
        # # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(input)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)  # state: size([8, 30]) or action: [32, 3] sum[length of list(admissible ids)] x MaxLen

        embedded = self.embedding(x_tt).permute(1,0,2)  # T x Batch x EmbDim size([30, 8, 128])
        return embedded


class StateRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, recurrent=False):
        super(StateRNN, self).__init__()
        # self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
        # self.h_ob = self.enc_ob.initHidden(self.batch_size)
        # self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        # self.h_look = self.enc_look.initHidden(self.batch_size)
        # self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        # self.h_inv = self.enc_inv.initHidden(self.batch_size)
        # vocab_size, embedding_dim, hidden_dim
        self.enc_look = PackedEncoderRNN(vocab_size, embedding_dim, hidden_dim)
        self.h_look = self.enc_look.initHidden(batch_size)
        self.enc_inv = PackedEncoderRNN(vocab_size, embedding_dim, hidden_dim)
        self.h_inv = self.enc_inv.initHidden(batch_size)
        self.enc_obs = PackedEncoderRNN(vocab_size, embedding_dim, hidden_dim)
        self.h_obs = self.enc_obs.initHidden(batch_size)
        self.enc_act = PackedEncoderRNN(vocab_size, embedding_dim, hidden_dim)
        self.h_act = self.enc_act.initHidden(batch_size)

        self.recurrent = recurrent
        self.to(device)

    def reset_hidden(self, done_mask_tt):
        '''
        Reset the hidden state of episodes that are done.
        :param done_mask_tt: Mask indicating which parts of hidden state should be reset.
        :return:
        '''
        self.h_look = done_mask_tt.detach() * self.h_look
        self.h_inv = done_mask_tt.detach() * self.h_inv
        self.h_obs = done_mask_tt.detach() * self.h_obs
        # self.h_act = done_mask_tt.detach() * self.h_act

    def clone_hidden(self):
        self.tmp_look = self.h_look.clone().detach()
        self.tmp_inv = self.h_inv.clone().detach()
        self.tmp_obs = self.h_obs.clone().detach()
        # self.tmp_act = self.h_act.clone().detach()

    def restore_hidden(self):
        self.h_look = self.tmp_look
        self.h_inv = self.tmp_inv
        self.h_obs = self.tmp_obs
        # self.h_act = self.tmp_act

    def forward(self, state):
        '''
        :param obs: Encoded observation tokens.
        :type obs: np.ndarray of shape (Batch_Size x 4 x 300)

        '''
        # Zip the state_batch into an easy access format
        state = State(*zip(*state))
        x_l, h_l = self.enc_look(state.description) #, self.h_look)
        x_i, h_i = self.enc_inv(state.inventory) #, self.h_inv)  # size([8, 128])
        x_o, h_o = self.enc_obs(state.obs) #, self.h_obs)

        if self.recurrent:
            self.h_look = h_l
            self.h_obs = h_o
            self.h_inv = h_i

        x = torch.cat((x_l, x_i, x_o), dim=1)
        h = torch.cat((h_l, h_i, h_o), dim=2)

        return x, h

    def forward_enc_act(self, action):
        x, _ = self.enc_act(action)
        return x


    def state_rnn(self, states):
        # Zip the state_batch into an easy access format
        state = State(*zip(*states))
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(state.obs, self.obs_encoder)
        look_out = self.packed_rnn(state.description, self.look_encoder)
        inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
        state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
        return state_out


class PackedEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PackedEncoderRNN, self).__init__()
        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # num_embeddings, embedding_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim)  # 128, 128

    def forward(self, input, hidden=None):
        lengths = torch.tensor([len(n) for n in input], dtype=torch.long, device=device)
        # # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(input)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort) # state: size([8, 30]) or action: [32, 3] sum[length of list(admissible ids)] x MaxLen

        embedded = self.embedding(x_tt).permute(1,0,2) # T x Batch x EmbDim size([30, 8, 128])

        # Pack the padded batch of sequences
        packed = pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False) # packed.data: size([240, 128])
        # packed = pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False) # embedded: size([300, 16, 100])
        if hidden is not None:
            output, hidden = self.gru(packed, hidden) # output: size([240, 128])
        else:
            output, hidden = self.gru(packed)
        # packed: [action] size([40, 128]) hidden: size([1, 8, 128]) next action 在这里出了问题?
        # Unpack the padded sequence
        output, _ = pad_packed_sequence(output) # output: size([30, 8,128])
        # output, _ = pad_packed_sequence(output)

        # Return only the last timestep of output for each sequence
        lengths=lengths.cuda()
        idx = (lengths-1).view(-1,1).expand(len(lengths), output.size(2)).unsqueeze(0)
        output = output.gather(0, idx).squeeze(0) # output: [state] size([8, 128]) [action] size([32, 128])
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = torch.zeros_like(e)
        zero_vec = zero_vec.fill_(9e-15)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x


class StateNetwork(nn.Module):
    def __init__(self, gat_emb_size, vocab, embedding_size, dropout_ratio, tsv_file, embeddings=None):
        super(StateNetwork, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embedding_size = embedding_size
        self.dropout_ratio = dropout_ratio
        self.gat_emb_size = gat_emb_size
        #self.params = params
        self.gat = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)

        self.pretrained_embeds = nn.Embedding(self.vocab_size, self.embedding_size)
        self.vocab_kge = self.load_vocab_kge(tsv_file)
        #self.init_state_ent_emb(params['embedding_size'])
        self.state_ent_emb = nn.Embedding.from_pretrained(torch.zeros((len(self.vocab_kge), self.embedding_size)), freeze=False)
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 100)

        self.to(device)

    def init_state_ent_emb(self, emb_size):
        embeddings = torch.zeros((len(self.vocab_kge), emb_size))
        for i in range(len(self.vocab_kge)):
            graph_node_text = self.vocab_kge[i].split('_')
            graph_node_ids = []
            for w in graph_node_text:
                if w in self.vocab.keys():
                    if self.vocab[w] < len(self.vocab) - 2:
                        graph_node_ids.append(self.vocab[w])
                    else:
                        graph_node_ids.append(1)
                else:
                    graph_node_ids.append(1)
            graph_node_ids = torch.LongTensor(graph_node_ids).cuda()
            cur_embeds = self.pretrained_embeds(graph_node_ids)

            cur_embeds = cur_embeds.mean(dim=0)
            embeddings[i, :] = cur_embeds
        self.state_ent_emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

    def load_vocab_kge(self, tsv_file):
        ent = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[int(eid.strip())] = e.strip()
        return ent

    def forward(self, graph_rep):
        out = []
        for g in graph_rep:
            node_feats, adj = g
            adj = torch.IntTensor(adj).cuda()
            x = self.gat.forward(self.state_ent_emb.weight, adj).view(-1)
            out.append(x.unsqueeze_(0))
        out = torch.cat(out)
        ret = self.fc1(out)
        return ret

# --------For BIDAF Part----------------------------------------
class BiAttention(nn.Module):
    def __init__(self, feature_size, dropout, single_att2cont=False, single_att2que=False):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(feature_size, 1, bias=False)
        self.memory_linear = nn.Linear(feature_size, 1, bias=False)
        self.dot_scale = nn.Parameter(
            torch.zeros(size=(feature_size,)).uniform_(1. / (feature_size ** 0.5)),
            requires_grad=True)
        self.init_parameters()
        self.single_att2cont = single_att2cont
        self.single_att2que = single_att2que

    def init_parameters(self):
        # xavier_uniform_(self.input_linear.weight.data, gain=0.1)
        # xavier_uniform_(self.memory_linear.weight.data, gain=0.1)
        return

    def forward(self, context, memory, mask):
        bsz, input_len = context.size(0), context.size(1)
        memory_len = memory.size(1)
        context = self.dropout(context)  # size([bsz*que_num, cont_len, 64])
        memory = self.dropout(memory)  # size([bsz*que_num, que_len, 64])

        input_dot = self.input_linear(context)  # size([32, cont_len, 1])
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)  # size([32, 1, que_len])
        cross_dot = torch.bmm(
            context * self.dot_scale,
            memory.permute(0, 2, 1).contiguous())  # size([32, 64, 2])
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])  # size([bsz*que_num, cont_len, que_len])

        weight_one = F.softmax(att, dim=-1)  # size([bsz*que_num, cont_len, que_len])
        output_one = torch.bmm(weight_one, memory)  # size([bsz*que_num, cont_len, 64])
        weight_two = (F.softmax(att.max(dim=-1)[0], dim=-1)  # size([bsz*que_num, 1, 64])
                      .view(bsz, 1, input_len))
        output_two = torch.bmm(weight_two, context)  # size([bsz*que_num, 1, 64])
        weight_three = (F.softmax(att.max(dim=1)[0], dim=-1)
                      .view(bsz, 1, memory_len))  # size([bsz*que_num, 1, que_len])
        output_three = torch.bmm(weight_three, memory)  # size([bsz*que_num, 1, 64])
        # return torch.cat(
        #     [context, output_one, context * output_one,
        #      output_two * output_one],
        #     dim=-1)
        if self.single_att2cont:
            return output_two.squeeze(1)  # size[bsz*que_num, 128])
        if self.single_att2que:
            return output_three.squeeze(1)
        else:
            return torch.cat([output_two, output_three], dim=-1).squeeze(1)  # size[bsz*que_num, 128])



class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        with torch.no_grad():
            m = (x.data.new(size=(x.size(0), 1, x.size(2)))
                 .bernoulli_(1 - dropout))
            mask = m.div_(1 - dropout)
            mask = mask.expand_as(x)
        return mask * x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat,
                 bidir, layernorm, return_last):
        super().__init__()
        self.layernorm = (layernorm == 'layer')
        if layernorm:
            self.norm = nn.LayerNorm(input_size)

        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(
                nn.GRU(input_size_, output_size_, 1,
                       bidirectional=bidir, batch_first=True))

        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(
                torch.zeros(size=(2 if bidir else 1, 1, num_units)),
                requires_grad=True) for _ in range(nlayers)])
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for rnn_layer in self.rnns:
                for name, p in rnn_layer.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(p.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(p.data)
                    elif 'bias' in name:
                        p.data.fill_(0.0)
                    else:
                        p.data.normal_(std=0.1)

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, inputs, input_lengths=None):
        bsz, slen = inputs.size(0), inputs.size(1)
        if self.layernorm:
            inputs = self.norm(inputs)
        output = inputs
        outputs = []
        outputs_last = []
        lens = 0
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            # output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens,
                                                  batch_first=True,
                                                  enforce_sorted=False)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:
                    # used for parallel
                    # padding = Variable(output.data.new(1, 1, 1).zero_())
                    padding = torch.zeros(
                        size=(1, 1, 1), dtype=output.type(),
                        device=output.device())
                    output = torch.cat(
                        [output,
                         padding.expand(
                             output.size(0),
                             slen - output.size(1),
                             output.size(2))
                         ], dim=1)
            if self.return_last:
                outputs_last.append(
                    hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2), torch.cat(outputs_last)
        return outputs[-1], outputs_last

# --------For BIDAF Part----------------------------------------
