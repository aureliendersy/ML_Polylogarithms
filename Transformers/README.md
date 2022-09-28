We use transformers to either

1) Simplify linear combinations of dilogarithms
2) Integrate symbols


### Data generation
For the first problem, the data generation is done in the `generate_starts.py` script of the RL folder.

For the second problem, the script `main_data.py` showcases an example of data generation for symbols with two entries. The data generated contains the file `data.mma` that is to be used by the Mathematica script `symbol_compute_script.nb` in order to compute the symbol. Once this is done both are to be merge together (also done in `main_data.py`) to create the file training set along the guidelines set out in https://github.com/facebookresearch/SymbolicMathematics.


### Model Training
The script `main_train.py` shows the parameters to be used when training the model, irrespective of the problem at hand. Models are trained using a single GPU with Apex optimization.

### Evaluation Run
The script `main_eval.py` shows the parameters to be used when evaluating the model. To get a complete picture of the performance of the model it is necessary to compute the symbol of the predicted outputs. This is done through an annex Mathematica script but with methods similar to `symbol_compute_script.nb`.
