The Reinforcement Learning approach to simplifying polylogarithms is centered around linear combinations of dilogarithms that simplify to 0, using a suitable choice of inversion, reflection or duplication.

### Data generation

The script `envs/generate_starts.py` is used to create linear combinations of dilogarithms. We start from a simple expression and then scramble it at random. The generator file allows control over the number of scrambles performed. 

The function `write_transformer_starts` will generate simple expressions and a data type relevant for the transformer models

The function `generate_null_starts_for_rl` will generate expressions that can be simplified down to 0 and which are useful for RL.


### Model Training

The script `run_script.py` is used to train the RL agents. 

There one can choose 
- The episode length and training time
- The type of reward function and its weights
- The size of the observation space and the nature of the embedding
- The list of actions allowed
- The dataset to train on
- The architecture for the policy and value network
- The RL agent and its hyperparameters
- The feature extractor used to embed the equations

### Model testing

The script `run_model.py` allows one to use the trained RL agent and evaluate its policy (retaining the top action choice). Detailed statistics are given, such as the action split, the number of steps taken, the solving rate as a function of complexity ...

The script `evaluate_algorithms.py` is used to simplify expressions contained in the test set by either using a classical algorithm (Modified Best First Search) or a trained RL agent (with beam search). The solving rate as a function of complexity is provided. 
