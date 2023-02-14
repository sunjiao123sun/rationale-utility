# rationale-utility
This is the repo for paper `Investigating the Utility of Free-Form Rationales`.

## What did I include?
As this is mostly an analysis work, I included two of the most important 
thing in this repository.
<ol>The four categories of rationales that we manually categorize, 
depending on if they leak the correct answer and if they contain additional 
background knowledge, necessary to answer the question.
</ol>
<ol>The scripts for the model training (i2o) and inference (i2r). The i2ro 
models can reuse the i2o model as rationales will be plugged into as the 
input in our setup. The basic model here is T5 model by SimpleT5. </ol>

This should be all you need to replicate our results, if you have questions, 
please reach out to Jiao Sun at jiaosun@usc.edu.