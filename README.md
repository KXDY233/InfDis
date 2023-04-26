# InfDis


## About

This is the source code for paper _Influence without Authority: Maximizing Information Coverage in Hypergraphs_ [SDM'23].



## Requirements

python>=3.3.7

multiprocessing

joblib

tqdm

This code was tested on Windows and Linux.

## Finding seeds

### Quick start

	python InfDis.py --data=email

### Parameters

`--data`: contact_primary, contact, email, w3cemail, geology, history, flickr, dblp, stackoverflow

`--probability`: independent propagation probability, default=0.01

### Save seeds

the seeds of all algorithms will be saved to the data path, e.g., "./data/email/".





## Evaluations

### Quick start

    python Evaluation.py --data=email --probability=0.01 --num_mcmc=100 --method=InfDis

### Parameters

`--data`: contact_primary, contact, email, w3cemail, geology, history, flickr, dblp, stackoverflow

`--probability`: independent propagation probability, default=0.01

`--earlystopping`: the number of steps of independent cascades

`--num_mcmc`: the number of mcmc simulations, default=100

`--method`: InfDis, Degree, Between, HyperRank


### Save results

the results will be saved to a fixed path, e.g., "./result/".


### Note

Multiprocessing is enabled by default, and **num_mcmc** should be larger than the number of CPU cores.





## Run Demo

please unzip "data.zip", and run:

    ./run_demo.bat

    ./run_demo.sh


## Cite

@inproceedings{li2023influence,
  title={Influence without Authority: Maximizing Information Coverage in Hypergraphs},
  author={Li, Peiyan and Wang, Honglian and Li, Kai and Bohm, Christian},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={10--18},
  year={2023},
  organization={SIAM}
}

