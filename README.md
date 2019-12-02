Machine Teaching of Active Sequential Learners
==============================================

This repository contains code and data implementing the methods and experiments described in

Tomi Peltola, Mustafa Mert Çelikok, Pedram Daee, Samuel Kaski  
**Machine Teaching of Active Sequential Learners**,  NeurIPS 2019  
https://papers.nips.cc/paper/9299-machine-teaching-of-active-sequential-learners  
(also available on arXiv: https://arxiv.org/abs/1809.02869)

There's also an accompanying website: https://aaltopml.github.io/machine-teaching-of-active-sequential-learners/

**Abstract**:  
Machine teaching addresses the problem of finding the best training data that can guide a learning algorithm to a target model with minimal effort. In conventional settings, a teacher provides data that are consistent with the true data distribution. However, for sequential learners which actively choose their queries, such as multi-armed bandits and active learners, the teacher can only provide responses to the learner's queries, not design the full data. In this setting, consistent teachers can be sub-optimal for finite horizons. We formulate this sequential teaching problem, which current techniques in machine teaching do not address, as a Markov decision process, with the dynamics nesting a model of the learner and the actions being the teacher's responses. Furthermore, we address the complementary problem of learning from a teacher that plans: to recognise the teaching intent of the responses, the learner is endowed with a model of the teacher. We test the formulation with multi-armed bandit learners in simulated experiments and a user study. The results show that learning is improved by (i) planning teaching and (ii) the learner having a model of the teacher. The approach gives tools to taking into account strategic (planning) behaviour of users of interactive intelligent systems, such as recommendation engines, by considering them as boundedly optimal teachers.


## Overview


### Requirements

 * Python 3.7
 * pytorch
 * pyro-ppl
 * numpy
 * scipy
 * scikit-learn
 * matplotlib
 * lifelines

You can create a conda environment satisfying the requirements with `conda env create -f environment.yml`.


### Running multi-armed bandit simulation experiments

Go to the `multi-armed_bandits` subfolder.

Example for running simulation experiments:  
`python target_sim.py wine 100 50 30 test`

This runs experiments on the wine dataset (data loaded from `simulation_studies/X_wine.npy`), with `100` arms, `50` replications with horizon of `30`. `test` gives an arbitrary id for the run used as part of the filename for the results (saved in `results` directory). Running the experiment can take some time (a few hours with the given settings on a relatively modern workstation).

To generate result figures from the saved experimental results, run  
`python plot_from_file.py experiment_wine_100_30_test`

Here, `experiment_wine_100_30_test` is the file name of the results file (without the `results` directory and file extension). The figures are generated in the `results` directory.

Multi-step (teacher's planning horizons of 1 to 4) experiments can be ran similarly using `target_sim_mla.py` and `plot_from_file_mla.py`.

Note: the results in the paper were ran on cluster computers with slightly different scripts (and different versions of dependencies), so results might not be reproduced exactly with the above even if all the settings are set to the same.


### Running active learning example and experiment

Go to the `active_learning` subdirectory.

Run `python example.py` to run the example.

Run `python experiment.py` to run the experiment.


### Code

The code in `multi-armed_bandits` subdirectory is organized as follows:

 * `target_sim.py`: Defines experimental settings and run replicate simulation experiments.
 * `dependent_arms_bandits.py`: Implements the experiment loop for one multi-armed bandit experiment with the given settings (called from `target_sim.py`).
 * `user_models.py`, `user_models_mix_obs.py`: User models implement the teaching behaviour.
 * `ai_models.py`, `ai_models_mix_obs.py`: AI models prepare data for user model's interpretation.
 * `logistic_regression_pyro.py`, `mixture_type_logistic_regression_pyro.py`, `logistic_regression_mla_pyro.py`: Implement the Pyro models and computation for posterior approximations.
 * `acquisition_functions.py`: Implements the bandit arm selection strategies.
 * `thompson_sampling_probabilities.py`: Implements the estimation of Thompson sampling probabilities.
 * `utils.py`: Wraps concordance computation code.
 * `user_study.py` and `user_study_evaluation.py`: Implements the scripts to run a user study. Result files of the user study in the paper are in `results_user_study` directory.


### Data

Pre-processed datasets are available in the `multi-armed_bandits/simulation_studies` directory (Wine data also in `active_learning/data`):

 * `X_word.npy` is the Word dataset [1].
 * `X_wine.npy` is the Wine dataset [2].
 * `X_leaf.npy` is the Leaf dataset [3].

Data for the user study is in `multi-armed_bandits/word_search_study` directory.

  [1] Distributed representations of words and phrases and their compositionality,
      Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff,
      Advances in Neural Information Processing Systems, NIPS, pages 3111--3119, 2013.
  
  [2] Modeling wine preferences by data mining from physicochemical properties,
      Cortez, Paulo and Cerdeira, António and Almeida, Fernando and Matos, Telmo and Reis, José
      Decision Support Systems, vol. 47, num. 4, pages=547--553, 2009.
      Dataset available at UCI ML Repository.
  
  [3] Evaluation of features for leaf discrimination,
      Silva, Pedro FB and Marcal, Andre RS and da Silva, Rubim M Almeida,
      International Conference on Image Analysis and Recognition, ICIAR, pages 197--204, 2013.
      Dataset available at UCI ML Repository.


## Contact

 * Tomi Peltola, tomi.peltola@aalto.fi
 * Mustafa Mert Çelikok, mustafa.celikok@aalto.fi
 * Pedram Daee, pedram.daee@aalto.fi

Work done in the [Probabilistic Machine Learning research group](https://research.cs.aalto.fi/pml/) at [Aalto University](https://www.aalto.fi/fi).


## Reference

 * Tomi Peltola, Mustafa Mert Çelikok, Pedram Daee, Samuel Kaski. **Machine Teaching of Active Sequential Learners**, NeurIPS 2019. https://papers.nips.cc/paper/9299-machine-teaching-of-active-sequential-learners


## License

GPL v3, see `LICENSE`
