# Instructions

0. Installation instructions: Make sure you have completed the instructions in the root folder of Spearmint's README.md


1. Fill out the part marked as TODO. Make sure your training is a blocking call (does not go into background)
2. make sure that you get wer and rtf at the end of your changes as described in rnnlm.py:train_rnnlm
3. Make sure you have followed the instructions in the main README.md 


4. Running instructions: Preferrably go to screen since this can run for some time.

```
cd Spearmint/spearmint
./cleanup.sh ../examples/rnnlm
# The above command cleans all records of the experiment in that folder (Governed by the experiment name in the config.json)

python main.py ../examles/rnnlm
```


# The above starts the constrained Bayesian Optimization step. 
# FAQs: 
1. How long will this run?
- This is configured to run for infinity. You must manually kill the job by ctrl + c

2. Common error: You may notice the iteration to increase every step very quickly, but with no results.
- This indicates a fault in the train_rnnlm method. Fix that, cleanup and rerun the experiment.


3. Where are the results?
- The outputs of each call is stored in ../examples/rnnlm/output, in the form of 0000000xx.out (xx is the job-id)

