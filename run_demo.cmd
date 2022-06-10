python InfDis.py     --data=contact
python Between.py    --data=contact
python Degree.py     --data=contact
python HyperRank.py  --data=contact
python Evaluation.py --data=contact --probability=0.01 --num_mcmc=100 --method=InfDis
python Evaluation.py --data=contact --probability=0.01 --num_mcmc=100 --method=Degree
python Evaluation.py --data=contact --probability=0.01 --num_mcmc=100 --method=Between
python Evaluation.py --data=contact --probability=0.01 --num_mcmc=100 --method=HyperRank
python Greedy.py     --data=contact --probability=0.01