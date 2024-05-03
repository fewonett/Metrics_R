
### Introduction:
In their paper “Generalized Random Forests” Athey et al. (2019) develop a general
framework for forest-based estimators, which allows for an unbiased nonparametric 
estimation of treatment effects as well as their variance. 
In this thesis, I will use the developed framework and the accompanying R package 
to investigate heterogenous treatment effects in the randomized control trial (RCT)
data by Carvalho et al. (2016).
At the basis of the framework are common random forests.
A random forest is based on many individual regression trees, each of which partitions
the covariate space into sub-groups. Every split is called a node and the final
subgroups are called leaves. The prediction for Y is generated as an average of
the observations falling into the same leaf.
The nonparametric nature of forests caters to the estimation of heterogenous treatment
effects. The conditional average treatment effects (E(treatment effect| X)) are 
not assumed to follow any defined functional form. In this context, causal forests
– one specific method derived from the generalized random forests (GRF) framework – 
will play an important role. In random forests, the optimal splits of the covariate
space when growing individual trees are typically chosen based on mean squared errors. Causal forests however, choose the splits based on treatment
effect heterogeneity: For each possible split, the treatment effects within the
partitioned groups are estimated, and subsequently, the split with the greatest 
possible difference in treatment effects is chosen. Additionally, a fraction of 
the data is kept aside while growing the forest, in order to calculate similarity 
scores between units. To generate treatment effect predictions Every new datapoint
is being sent through each tree in the forest, and similarity scores are then 
calculated based on how often the datapoint ends  up in the same leaves as the 
training observations. The similarity scores are then used as weights, when calculating 
the treatment effect predictions as a weighted average of the treatment effects 
of similar datapoints in the training data. This way an individual treatment effect
prediction conditional on the covariate values can be generated for every datapoint.
In section 1 I will explore certain properties of random and causal forests.For 
an in-detail discussion of the algorithms see section 6.2 of Athey et al. (2019)
and the algorithmic reference section of the grf documentation (2023).
My code will be structured into two main sections:
In the first section, I will introduce the reader to the functionalities and syntax
of the grf package. The parameters of the main functions will be explained as well.
A synthetic dataset is created and used to apply the explained theory. 
Once a basic understanding of the package is developed, the heterogeneity analysis
of real data from Carvalho et al. (2016) begins in section 2.
In a first step, the data is prepared for usage in a causal forest. Then in section
2.1, average and conditional average treatment effects are calculated using a causal
forest. In a first effort to grasp treatment effect heterogeneity, the best linear
projection of the treatment effects based on the covariates is retrieved in section
2.2: Assuming linearity, by how much does the conditional average treatment effect
in- or decrease in each covariate?
In section 2.3 I turn towards the main task of the thesis: Recreating figure 9
of Carter et al. (2019) to answer the question: How do treatment effects change
in one covariate while all other covariates are held constant at a certain percentile?
In section 2.4 and 2.5 I explore possibility of targeting high-treatment-effect
units. First, I use the rank-weighted average treatment (Yadlowsky et al. (2021)) 
effect which helps to evaluate possible targeting rules derived from a causal forest.
Second, the package policytree (after Sverdrup et al. (2019)) is used to create 
a tree-based rule in order to target units with very high or low treatment effects.
Throughout the thesis explanatory comments are added both with regards to econometric
theory and the code.

### References in scientific literature:
Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.  
Carter, M. R., Tjernström, E., & Toledo, P. (2019). Heterogeneous impact dynamics of a rural business development program in Nicaragua. Journal of Development Economics, 138, 77-98.  
Carvalho, L. S., Meier, S., & Wang, S. W. (2016). Poverty and economic decision-making: Evidence from changes in financial resources at payday. American economic review, 106(2), 260-284.  
Farbmacher, H., Kögel, H., & Spindler, M. (2021). Heterogeneous effects of poverty on attention. Labour Economics, 71, 102028.  
Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., & Wager, S. (2020). policytree: Policy learning via doubly robust empirical welfare maximization over trees. Journal of Open Source Software, 5(50), 2232.  
Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2021). Evaluating treatment prioritization rules via rank-weighted average treatment effects. arXiv preprint arXiv:2111.07966.  

Other references:
GRF package documentation (2023): https://grf-labs.github.io/grf/index.html  
GRF Github repository (2023): https://github.com/grf-labs/grf  
Policytree documentation (2022): https://grf-labs.github.io/policytree/articles/policytree.html   
