"""
Contains the models that will be used as the feature extractors to handle the encoding of the sentences
Contains the GNN usefull for generating the sentence embeddings
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
import torch.nn as nn
import math
import torch
from torch_geometric.nn import GCN, GraphSAGE, GAT, global_add_pool
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch import Tensor
import gym
import torch as th


class PositionalEncoding(nn.Module):
    """
    If we want to generate a positional encoding of the tokens using sinusoidal embeddings
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoding(BaseFeaturesExtractor):
    """
    If we want to apply a transformer encoder
    """

    def __init__(self, observation_space: gym.spaces.Box, max_sent_length=512, embed_dim=256, n_words=30, num_heads=4,
                 num_layers=3):
        self.max_sentence_length = max_sent_length
        self.embedding_dim = embed_dim
        self.number_words = n_words
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = self.embedding_dim * 4
        super(TransformerEncoding, self).__init__(observation_space, self.embedding_dim*self.max_sentence_length)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_type = 'Transformer encoding'
        self.position_embeddings = PositionalEncoding(d_model=self.embedding_dim, max_len=self.max_sentence_length)
        self.embeddings = nn.Embedding(self.number_words, self.embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=self.embedding_dim ** -0.5)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads,
                                                        dim_feedforward=self.hidden_dim, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        src_mask = torch.triu(torch.ones(self.max_sentence_length, self.max_sentence_length) * float('-inf'),
                              diagonal=1).to(self.device)
        observations_in = observations.transpose(0, 1).long()
        tensor = self.embeddings(observations_in) * math.sqrt(self.embedding_dim)
        pos_tensor = self.position_embeddings(tensor)
        encoded = self.transformer_encoder(pos_tensor, src_mask)

        return encoded.transpose(0, 1).flatten(start_dim=1)


class RNNEncoder(BaseFeaturesExtractor):
    """
    To apply a generic RNN encoder
    """
    def __init__(self, observation_space: gym.spaces.Box, embed_dim=64, num_layers=2, n_words=30, bidirectional=True):
        self.embedding_dim = embed_dim
        self.hidden_dim = self.embedding_dim * 4
        self.num_layers = num_layers
        self.number_words = n_words
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_dim = self.hidden_dim*2 if bidirectional else self.embedding_dim
        super(RNNEncoder, self).__init__(observation_space, out_dim)
        self.embeddings = nn.Embedding(self.number_words, self.embedding_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs_np = observations.detach().cpu().numpy()
        max_length = max([len(batch_el[batch_el > 0]) for batch_el in obs_np])
        observations_in = observations[:, : max_length].long()
        embed_input = self.embeddings(observations_in)

        return embed_input


class GRUEncoder(RNNEncoder):
    """
    Apply a GRU encoder layer
    """

    def __init__(self, observation_space: gym.spaces.Box, embed_dim=64, num_layers=2, n_words=30, bidirectional=True):
        super(GRUEncoder, self).__init__(observation_space, embed_dim=embed_dim, num_layers=num_layers, n_words=n_words,
                                         bidirectional=bidirectional)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True,
                          bidirectional=self.bidirectional)
        self.model_type = 'GRU encoding'

    def forward(self, observations: th.Tensor) -> th.Tensor:
        embed_input = super(GRUEncoder, self).forward(observations)
        rnn_embed, _ = self.rnn(embed_input)

        # Return the last temporal state of the GRU as the sentence embedding
        return rnn_embed[:, -1, :]


class LSTMEncoder(RNNEncoder):

    """
    Apply a LSTM encoder layer
    """
    def __init__(self, observation_space: gym.spaces.Box, embed_dim=64, num_layers=2, n_words=30, bidirectional=True):
        super(LSTMEncoder, self).__init__(observation_space, embed_dim=embed_dim, num_layers=num_layers,
                                          n_words=n_words, bidirectional=bidirectional)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True,
                           bidirectional=self.bidirectional)
        self.model_type = 'LSTM encoding'

    def forward(self, observations: th.Tensor) -> th.Tensor:
        embed_input = super(LSTMEncoder, self).forward(observations)
        rnn_embed, _ = self.rnn(embed_input)

        # Return the max pooling of the LSTM over the whole hidden states
        return torch.max(rnn_embed, 1)[0]


class GraphEncoder(BaseFeaturesExtractor):

    """
    Apply a GNN as an initial encoder layer
    """
    def __init__(self, observation_space: gym.spaces.Box, obs_space=None, embed_dim=64, num_layers=2,
                 bidirectional=True):

        # Model parameters
        self.embedding_dim = embed_dim
        self.obs_space = obs_space
        self.number_words = len(obs_space.words)
        self.num_layers = num_layers

        # For an undirected graph
        self.bidirectional = bidirectional

        if self.obs_space is None:
            raise TypeError('Need an observation space here')
        out_dim = 2*self.embedding_dim

        super(GraphEncoder, self).__init__(observation_space, out_dim)

        # Proper handling of device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If we don't one hot encode than we add a dedicated embedding layer
        if not self.obs_space.one_hot_encode:
            self.embeddings = nn.Embedding(self.number_words, self.embedding_dim)
            self.input_size = self.embedding_dim
        else:
            self.embeddings = nn.Identity()
            self.input_size = self.number_words

        # Use two networks - one for the full equation and one for the term being attended to by the agent
        self.gcn_forward = None
        self.gcn_fwd_state = None
        self.model_type = 'Graph encoding'

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass
        :param observations:
        :return:
        """

        # Construct the relevant graphs using the observation data (i.e the equation)
        obs_np = observations.detach().cpu().numpy()
        graph_infos = [self.obs_space.obs_to_graph(ind_obs_np) for ind_obs_np in obs_np]
        graph_first_term = [self.obs_space.extract_graph_first_term(graph_info[0], graph_info[1])
                            for graph_info in graph_infos]

        # For an undirected graph
        if self.bidirectional:
            graphs = [Data(x=self.embeddings(torch.tensor(graph_info[0], dtype=torch.float, device=self.device)),
                           edge_index=to_undirected(torch.tensor(graph_info[1], dtype=torch.long,
                                                                 device=self.device).t().contiguous()))
                      for graph_info in graph_infos]
            graphs_first = [Data(x=self.embeddings(torch.tensor(graph_frt[0], dtype=torch.float, device=self.device)),
                                 edge_index=to_undirected(torch.tensor(graph_frt[1], dtype=torch.long,
                                                                       device=self.device).t().contiguous()))
                            for graph_frt in graph_first_term]
        # For a directed graph
        else:
            graphs = [Data(x=self.embeddings(torch.tensor(graph_info[0], dtype=torch.float, device=self.device)),
                           edge_index=torch.tensor(graph_info[1], dtype=torch.long,
                                                   device=self.device).t().contiguous()) for graph_info in graph_infos]
            graphs_first = [Data(x=self.embeddings(torch.tensor(graph_frt[0], dtype=torch.float, device=self.device)),
                            edge_index=torch.tensor(graph_frt[1], dtype=torch.long,
                                                    device=self.device).t().contiguous())
                            for graph_frt in graph_first_term]

        # Can proceed everything in batches
        batch_size = len(graphs)
        dataloader_graph = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        big_graph = next(iter(dataloader_graph))
        batch_vect = big_graph.batch

        dataloader_graph_f = DataLoader(graphs_first, batch_size=batch_size, shuffle=False)
        big_graph_f = next(iter(dataloader_graph_f))
        batch_vect_f = big_graph_f.batch

        forwards = self.gcn_forward(big_graph.x, big_graph.edge_index)
        forwards_fst = self.gcn_fwd_state(big_graph_f.x, big_graph_f.edge_index)

        # Add a final pooling layer to add the final embeddings
        pooled = global_add_pool(forwards, batch_vect)
        pooled_fst = global_add_pool(forwards_fst, batch_vect_f)

        # Return the concatenation of the full equation embedding and the term being attended to
        return torch.cat((pooled, pooled_fst), dim=1)


class GCNEncoder(GraphEncoder):
    """
    Graph convolutional network
    """
    def __init__(self, observation_space: gym.spaces.Box, obs_space=None, embed_dim=64, num_layers=2,
                 bidirectional=True):
        super(GCNEncoder, self).__init__(observation_space, obs_space, embed_dim, num_layers, bidirectional)

        self.gcn_forward = GCN(in_channels=self.input_size, hidden_channels=self.embedding_dim,
                               num_layers=self.num_layers, aggr='mean')
        self.gcn_fwd_state = GCN(in_channels=self.input_size, hidden_channels=self.embedding_dim,
                                 num_layers=self.num_layers, aggr='mean')
        self.model_type = 'GCN Graph encoding'


class GSageEncoder(GraphEncoder):
    """
    Graph Sage network
    """
    def __init__(self, observation_space: gym.spaces.Box, obs_space=None, embed_dim=64, num_layers=2,
                 bidirectional=True):
        super(GSageEncoder, self).__init__(observation_space, obs_space, embed_dim, num_layers, bidirectional)

        self.gcn_forward = GraphSAGE(in_channels=self.input_size, hidden_channels=self.embedding_dim,
                                     num_layers=self.num_layers, aggr='mean')
        self.gcn_fwd_state = GraphSAGE(in_channels=self.input_size, hidden_channels=self.embedding_dim,
                                       num_layers=self.num_layers, aggr='mean')
        self.model_type = 'GSage Graph encoding'


class GATEncoder(GraphEncoder):

    def __init__(self, observation_space: gym.spaces.Box, obs_space=None, embed_dim=64, num_layers=2,
                 bidirectional=True):
        super(GATEncoder, self).__init__(observation_space, obs_space, embed_dim, num_layers, bidirectional)

        self.gcn_forward = GAT(in_channels=self.input_size, hidden_channels=self.embedding_dim,
                               num_layers=self.num_layers, aggr='mean')
        self.gcn_fwd_state = GAT(in_channels=self.input_size, hidden_channels=self.embedding_dim,
                                 num_layers=self.num_layers, aggr='mean')
        self.model_type = 'GAT Graph encoding'


class EmbedEncoder(BaseFeaturesExtractor):
    """
    Dedicated token embedding layer
    """
    def __init__(self, observation_space: gym.Space,  embed_dim=64, n_words=30, max_sent_length=512):
        self.embedding_dim = embed_dim
        self.max_sentence_length = max_sent_length
        self.number_words = n_words
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(EmbedEncoder, self).__init__(observation_space, self.embedding_dim*self.max_sentence_length)
        self.embeddings = nn.Embedding(self.number_words, self.embedding_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        embed_input = self.embeddings(observations.long())
        return self.flatten(embed_input)


class CombinedFeatureExtractor(BaseFeaturesExtractor):
    """Wrapper to handle the connection with stable baselines"""
    def __init__(self, observation_space: gym.spaces.Dict, class_extract=None, kwargs_extract=None):

        if kwargs_extract is not None and 'positional' in kwargs_extract:
            if kwargs_extract.get("positional"):
                self.positional = True
            kwargs_extract.pop('positional')

        else:
            self.positional = False

        # If we get additional parameters we have to pass them
        if kwargs_extract is None:
            word_encoder = class_extract(observation_space['words'])
        else:
            word_encoder = class_extract(observation_space['words'], **kwargs_extract)

        # Ensure the proper device is used
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If we add the additional information (e.g last action) then we have to add it to the vector
        dim_add = observation_space['prev_state'].shape[0]
        max_len, dict_size = observation_space.spaces['words'].shape
        super(CombinedFeatureExtractor, self).__init__(observation_space, word_encoder._features_dim + dim_add)

        self.word_encoder = word_encoder

        # Can add positional encoding if we want
        if kwargs_extract is not None and 'positional' in kwargs_extract and kwargs_extract.get("positional"):
            self.pos_encod = True
            self.pos_encoder = nn.Embedding(max_len, dict_size)
            self.layer_norm_emb = nn.LayerNorm(dict_size, eps=1e-12)
            nn.init.normal_(self.pos_encoder.weight, mean=0, std=dict_size ** -0.5)
        else:
            self.pos_encod = False

    def forward(self, observations) -> th.Tensor:
        """ Forward pass"""
        words = observations['words']

        if self.pos_encod:
            positions = torch.arange(words.size()[1], out=words.new(words.size()[1]).long()).unsqueeze(0)
            words = words + self.pos_encoder(positions)
            words = self.layer_norm_emb(words)

        # Use the encoder (e.g GNN)
        encoded_words = self.word_encoder(words)

        # Normalize the extra information and add it to the observation vector
        prev_state = observations['prev_state']
        low_bound = self._observation_space['prev_state'].low
        high_bound = self._observation_space['prev_state'].high
        encoded_prev_state = (prev_state.cpu()-low_bound)/(high_bound-low_bound)
        return th.cat((encoded_words, encoded_prev_state.float().to(self.device)), dim=1)

