"""
Here we define the different constituents necessary to make the RL algorithms work
"""

from abc import abstractmethod
from logging import getLogger
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from .feature_extractor import TransformerEncoding, LSTMEncoder, GRUEncoder, EmbedEncoder,\
    CombinedFeatureExtractor, GSageEncoder, GATEncoder, GCNEncoder
from stable_baselines3.common.torch_layers import FlattenExtractor
import torch as th
from gym import spaces

logger = getLogger()

# Dictionary for mappings
map_layers = {'relu': th.nn.ReLU, 'tanh': th.nn.Tanh, 'sigmoid': th.nn.Sigmoid, 'leaky': th.nn.LeakyReLU}
map_extractors = {'transformer': TransformerEncoding, 'gru': GRUEncoder, 'lstm': LSTMEncoder, 'graph': GCNEncoder,
                  'embed': EmbedEncoder, 'gcn': GCNEncoder, 'gat': GATEncoder, 'sage': GSageEncoder}


class RLModel:
    """Core class for defining the RL model"""
    def __init__(self, environment, policy, value_layers, policy_layers, activation_func,
                 verbose=1, feature_extractor=None, seed=0, one_hot=False):

        self.verbose = verbose

        self.policy = policy
        self.policy_kwargs = None
        self.value_layers = value_layers
        self.policy_layers = policy_layers
        self.activation_func = activation_func

        self.environment = environment
        self.one_hot = one_hot
        self.seed = seed
        self.feature_extractor = feature_extractor
        self.check_policy()

        if self.one_hot and not self.feature_extractor:
            raise ValueError('If we one hot encode, we need a feature extractor')

        if isinstance(self.feature_extractor, dict) and self.one_hot\
                and self.feature_extractor['name'] in ['gru', 'lstm', 'transformer', 'embed']:
            raise ValueError('If we use transformer/LSTM encoding, we do not one hot encode')

        if self.activation_func not in map_layers.keys():
            raise NameError('{} not registered here'.format(self.activation_func))

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def build_net_arch(self):
        raise not NotImplementedError

    @abstractmethod
    def get_name(self):
        raise NotImplementedError

    def check_policy(self):
        if isinstance(self.environment.observation_space, spaces.Dict):
            if self.policy == 'MlpPolicy':
                logger.warning('Policy passed was MlpPolicy but observations are dict. Defaulting to MultiInputPolicy')
                self.policy = 'MultiInputPolicy'


class OnPolicyModel(RLModel):
    """
    Wrapper for On Policy RL Models
    """
    def __init__(self, environment, policy, shared_layers, value_layers, policy_layers, activation_func,
                 verbose=1, feature_extractor=None, seed=0, one_hot=False):
        super(OnPolicyModel, self).__init__(environment, policy, value_layers, policy_layers, activation_func,
                                            verbose, feature_extractor, seed=seed, one_hot=one_hot)
        self.shared_layers = shared_layers
        self.build_net_arch()

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def build_net_arch(self):
        """
        Given the parameters we have we build the architecture for the policy and value nets
        :return:
        """

        # Start by building the policy and value layers - can be shared
        dict_arch = {}
        if self.value_layers is not None:
            dict_arch['vf'] = self.value_layers
            logger.info('Value network has {} layers'.format(str(self.value_layers)))
        if self.policy_layers is not None:
            dict_arch['pi'] = self.policy_layers
            logger.info('Policy network has {} layers'.format(str(self.policy_layers)))

        if self.shared_layers is not None:
            net_arch = self.shared_layers if not dict_arch else self.shared_layers + [dict_arch]
            logger.info('Policy and value networks share {} layers'.format(str(self.shared_layers)))
        else:
            net_arch = [dict_arch]

        dict_params = dict(activation_fn=map_layers[self.activation_func], net_arch=net_arch)

        # If we have an extra feature extractor in front (e.g a GNN)
        if isinstance(self.feature_extractor, dict):
            class_extract = map_extractors[self.feature_extractor['name']]
            extract_kwargs = self.feature_extractor['params']
        else:
            class_extract = FlattenExtractor
            extract_kwargs = None

        # If we need to add extra info to the observation vector (e.g the last action used)
        if isinstance(self.environment.obs_rep.space, spaces.Dict):
            dict_params['features_extractor_class'] = CombinedFeatureExtractor
            params_extract = {'class_extract': class_extract, 'kwargs_extract': extract_kwargs}
            dict_params['features_extractor_kwargs'] = params_extract

        else:
            dict_params['features_extractor_class'] = class_extract
            if extract_kwargs is not None:
                dict_params['features_extractor_kwargs'] = extract_kwargs

        self.policy_kwargs = dict_params

    @abstractmethod
    def get_name(self):
        raise NotImplementedError


class TRPOModel(OnPolicyModel):
    """TRPO model from stable baselines 3"""
    def __init__(self,  environment, policy, shared_layers, value_layers, policy_layers, activation_func,
                 feature_extractor=None, seed=0, one_hot=False, **kwargs):

        if feature_extractor is not None and isinstance(feature_extractor, dict)\
                and feature_extractor['name'] == 'lstm':
            raise TypeError("Cannot use LSTM with TRPO that requires a double backwards")

        super(TRPOModel, self).__init__(environment, policy,
                                        shared_layers, value_layers, policy_layers, activation_func,
                                        feature_extractor=feature_extractor, seed=seed, one_hot=one_hot)
        self.trpo_kwargs = kwargs

    def get_model(self):
        return TRPO(self.policy, self.environment, policy_kwargs=self.policy_kwargs, verbose=self.verbose,
                    seed=self.seed, **self.trpo_kwargs)

    def get_name(self):
        return 'trpo'


class PPOModel(OnPolicyModel):
    """PPO model from stable baselines 3"""
    def __init__(self, environment, policy, shared_layers, value_layers, policy_layers, activation_func,
                 feature_extractor=None, seed=0, one_hot=False, **kwargs):
        super(PPOModel, self).__init__(environment, policy,
                                       shared_layers, value_layers, policy_layers, activation_func,
                                       feature_extractor=feature_extractor, seed=seed, one_hot=one_hot)
        self.ppo_kwargs = kwargs

    def get_model(self):
        return PPO(self.policy, self.environment, policy_kwargs=self.policy_kwargs, verbose=self.verbose,
                   seed=self.seed, **self.ppo_kwargs)

    def get_name(self):
        return 'ppo'
