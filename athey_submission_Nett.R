# Econometrics in R submission:
# Estimating heterogenous treatment effects using the generalized random forest framework

# by Fewo Nett

# Introduction:
# In their paper “Generalized Random Forests” Athey et al. (2019) develop a general
# framework for forest-based estimators, which allows for an unbiased nonparametric 
# estimation of treatment effects as well as their variance. 
# In this thesis, I will use the developed framework and the accompanying R package 
# to investigate heterogenous treatment effects in the randomized control trial (RCT)
# data by Carvalho et al. (2016).
# At the basis of the framework are common random forests.
# A random forest is based on many individual regression trees, each of which partitions
# the covariate space into sub-groups. Every split is called a node and the final
# subgroups are called leaves. The prediction for Y is generated as an average of
# the observations falling into the same leaf.
# The nonparametric nature of forests caters to the estimation of heterogenous treatment
# effects. The conditional average treatment effects (E(treatment effect| X)) are 
# not assumed to follow any defined functional form. In this context, causal forests
# – one specific method derived from the generalized random forests (GRF) framework – 
# will play an important role. In random forests, the optimal splits of the covariate
# space when growing individual trees are chosen based on mean squared errors (or 
# in some cases entropy). Causal forests however, choose the splits based on treatment
# effect heterogeneity: For each possible split, the treatment effects within the
# partitioned groups are estimated, and subsequently, the split with the greatest 
# possible difference in treatment effects is chosen. Additionally, a fraction of 
# the data is kept aside while growing the forest, in order to calculate similarity 
# scores between units. To generate treatment effect predictions Every new datapoint
# is being sent through each tree in the forest, and similarity scores are then 
# calculated based on how often the datapoint ends  up in the same leaves as the 
# training observations. The similarity scores are then used as weights, when calculating 
# the treatment effect predictions as a weighted average of the treatment effects 
# of similar datapoints in the training data. This way an individual treatment effect
# prediction conditional on the covariate values can be generated for every datapoint.
# In section 1 I will explore certain properties of random and causal forests.For 
# an in-detail discussion of the algorithms see section 6.2 of Athey et al. (2019)
# and the algorithmic reference section of the grf documentation (2023).
# My code will be structured into two main sections:
# In the first section, I will introduce the reader to the functionalities and syntax
# of the grf package. The parameters of the main functions will be explained as well.
# A synthetic dataset is created and used to apply the explained theory. 
# Once a basic understanding of the package is developed, the heterogeneity analysis
# of real data from Carvalho et al. (2016) begins in section 2.
# In a first step, the data is prepared for usage in a causal forest. Then in section
# 2.1, average and conditional average treatment effects are calculated using a causal
# forest. In a first effort to grasp treatment effect heterogeneity, the best linear
# projection of the treatment effects based on the covariates is retrieved in section
# 2.2: Assuming linearity, by how much does the conditional average treatment effect
# in- or decrease in each covariate?
# In section 2.3 I turn towards the main task of the thesis: Recreating figure 9
# of Carter et al. (2019) to answer the question: How do treatment effects change
# in one covariate while all other covariates are held constant at a certain percentile?
# In section 2.4 and 2.5 I explore possibility of targeting high-treatment-effect
# units. First, I use the rank-weighted average treatment (Yadlowsky et al. (2021)) 
# effect which helps to evaluate possible targeting rules derived from a causal forest.
# Second, the package policytree (after Sverdrup et al. (2019)) is used to create 
# a tree-based rule in order to target units with very high or low treatment effects.
# Throughout the thesis explanatory comments are added both with regards to econometric
# theory and the code.



# This code block identifies which of the required packages are not yes installed
# and subsequently loads them:
list.of.packages <- c("grf", "ggplot2", "zoo", "dplyr", "policytree", "png", "grid", "gridExtra", "scales", "DiagrammeR")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if (length(new.packages)) install.packages(new.packages)

# Restart the R session once after running the installs!

################################################################################

# Before running the code please update the working directory to the respective folder
# on your machine. Place the dataset in the same folder.
#setwd("SET_YOUR_PATH_HERE")

library(grf)
library(ggplot2)
library(zoo)
library(dplyr)
library(policytree)
library(png)
library(grid)
library(gridExtra)
library(scales)

set.seed(110)

# SECTION 1: Introduction to the grf package:

# Define functions of Section 1
# Note: I am aware that typically all functions are defined at the beginning of the 
# script. However, I will clear the environment at the end of some sections to keep
# it orderly. Therefore, I will define functions at the beginning of sections.

# Define a function to calculate the mean squared error from a vector with individual
# errors:
mse <- function(vec) {
  # Raise error if the input is not of the right format
  if (class(vec) != "numeric") {
    stop("Input must be numeric vector")
  }
  # The return value:
  mean(vec^2, na.rm = TRUE)
}

# Create a synthetic dataset:
n <- 5000
p <- 10
X <- 100 * matrix(rnorm(n * p), n, p)

# W is the treatment indicator, which will be relevant when dealing with causal
# forests later:
W <- rbinom(n, 1, 0.5)

# We will create our variable of interest Y as a function of W and the first 5 columns
# of X:
Y <- pmax(X[, 1], 0) * W + X[, 2] + 3 * pmin(X[, 3], 0) + pmin(X[, 4], 0) - 3 * X[, 5] + rnorm(n)

# For the moment we will include the treatment indicator W as a normal variable:
X <- cbind(X, W)

# Now we will try to predict Y with a random forest. The function implementing "normal"
# random forests in the grf package is called regression_forest.
r.forest <- regression_forest(X, Y)

# We can retrieve the predicted Y values for each observation with predict():
Y.hat <- predict(r.forest, estimate.variance = TRUE)

# Let's take a closer look at the main parameters in the regression_forest function.
# First, there is the number of trees in the random forest, which defaults to 2000:
regression_forest(X, Y, num.trees = 10)

# Note that we can plot individual trees from the forest. However, the trees are
# very deep and therefore hard to understand visually:
# Get the first tree of the forest
tree <- get_tree(r.forest, 1)
# Plot it:
plot(tree)


# As discussed in the lecture, an individual regression tree is unbiased in expectation
# but that does not necessarily mean it will perform well in prediction. Individual
# trees are very sensitive to sample selection: Adding or removing individual datapoints
# can lead to vastly different predictions.
# Random forests address these problems in several ways.
# 1. They average over the predictions of all trees in the forest. For each X observation
# a prediction of Y is made by all trees, and subsequently an average is taken.
# We can see how the predictions converge towards the true Y as the number of trees
# increases.
# We will now run the same random forest many times with an increasing number of
# trees and save the mean squared error each time.

# To reduce the runtime we will not compute the random forest  for every tree number.
# n sets the increment by which we increase the tree number each time:
n <- 2
count <- 0

# Lets create a dataframe to store the mean squared error and tree 
# number, and runtime of each forest:
errors <- data.frame(tree_number = numeric(), mean_squared_error = numeric(), runtime = numeric())

#  WARNING: This loop will grow many random forests in a row which can take a few
# minutes and use a lot of RAM. You can increase n above to reduce the workload.
# However, the graph will be less pretty.
min <- 5
max <- 500
for (treenum in min:max) {
  # Incrementing counter, to determine whether we should compute the forest or not:
  count <- count + 1
  # Grow the random forest only if the count is divisible by n:
  if (count %% n == 0) {
    start_time <- Sys.time()
    r.forest <- regression_forest(X, Y, num.trees = treenum)
    Y.hat <- predict(r.forest)
    end_time <- Sys.time()
    # Calculate the Mean Squared Error
    mean_squared_error <- mse(Y - Y.hat$predictions)
    # Append the tree number and the mean squared error to the dataframe
    row <- c(treenum, mean_squared_error, end_time - start_time)
    print(paste0(paste0(as.character(treenum), "/"), as.character(max)))
    errors[nrow(errors) + 1, ] <- row
  }
}

# Plot the mean squared error as a function of the number of trees
ggplot(errors, aes(x = tree_number, y = mean_squared_error)) +
  geom_point(size = 0.8) +
  geom_smooth(method = "loess", span = 0.3, se = FALSE, linewidth = 0.5) +
  xlab("Number of Trees") +
  ylab("Mean Squared Error") +
  ggtitle("Prediction accuracy and the number of trees", ) +
  theme(plot.title = element_text(hjust = 0.5))
# As you can see the the mean squared error is shrinking in the number of trees.

# More trees do improve performance but they also increase runtime:
ggplot(errors, aes(x = tree_number, y = runtime)) +
  geom_point(size = 0.7) +
  geom_smooth(method = "lm", formula = y ~ x, color = "red", se = FALSE, linewidth = 0.5) +
  xlab("Number of Trees") +
  ylab("Runtime") +
  ggtitle("Runtime and the number of trees", ) +
  theme(plot.title = element_text(hjust = 0.5))
# This is one of the trade offs to be considered when using forest based algorithms.


# 2.The second way random forests address the problems of individual regression trees
# is by growing each tree in the forest only on a random subsample of the available 
# observations. We can adjust the size of these subsamples with another hyperparameter 
# sample.fraction.

# sample.fraction thus defines how big the random sample to grow each tree is:
r.forest <- regression_forest(X, Y, num.trees = 100, sample.fraction = 0.3)
# The same principle is applied to the number of covariates used to grow trees. 
# Especially in high dimensional data (large number of covariates) it is common 
# to grow individual trees only on a random ubset of the covariates as well.

# We define the number of covariates used in each tree via the parameter mtry:
r.forest <- regression_forest(X, Y, num.trees = 150,
                              mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)))
# The shown value  mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)) is the package
# default: This means by default all columns are used if the number of columns in
# X is smaller than 20 plus the square root of the column number.

# As discussed in the lecture it is generally not a good idea to evaluate the performance
# of a tree based model on the data it was trained  on. However, this is exactly 
# what we did so far.
# Note that predict(r.forest) returns y estimates for every observation of covariates
# in the training data unless it is passed another dataset explicitly (which I did
# not until now).The standard
# way of addressing the problem of overfitting is by dividing the dataset into
# a training set and an evaluation set, which can be done as follows:

# Divide the sample by 2, by using the list train:
train <- 1:(nrow(X) / 2)

# Train the regression forest only on the training half of the data:
r.forest <- regression_forest(X[train, ], Y[train], num.trees = 150)
# Note that X[train,] and Y[train] will return the first half of the data each.

# Now we predict the Y values for the other half of the data
Y.hat <- predict(r.forest, X[-train, ], estimate.variance = TRUE)

# Now we can manually check how good the predictions perform, because we know the
# true value for the predicted ys:
errors <- Y.hat[, "predictions"] - Y[-train]

# The vector errors now contains the errors for each observation:
head(errors)
mse(errors)
# From here we can go on and calculate the mean squared error or estimate the variance.
# However, predict() will return a variance estimate if we set estimate.variance
# to be true:
head(Y.hat)

# Previously we never proceeded in this way (separating training and evaluation
# data). The reason we can ignore this without overfitting, is that the grf package 
# supports "out of bag prediction".
# Remember that random forests grow each tree using only a subsample of all observations.
# Essentially the package "remembers" which observations were used to grow which
# trees and subsequently only calculates the predictions for each tree based on
# observations which were not used to grow it. Intuitively a tree cannot be overfitted
# on data that it was not trained on. Athey et al. (2019) showed formally that this
# leads to unbiased estimation of Y.

# Before moving on to causal forests I want to discuss one particularly powerful
# feature: Parameter tuning.
# Up until now a few important parameters were discussed: The number of
# trees, the sample fraction, and mtry. How does one decide which values to choose?
# Of course it is possible to manually investigate the relation between parameter
# value, error rates, and runtime like previously shown for the number of trees,
# but this is highly work intensive. Also keep in mind the parameters do not necessarily
# influence model performance independently of each other which makes this a complex
# multidimensional optimization problem. Luckily, the package offers an option for
# automated parameter tuning, which approximates an optimal parameter combination:

# Tune all parameters:
r.forest <- regression_forest(X, Y, tune.parameters = "all")
# Note that this process can take very long for bigger datasets, since it essentially
# uses brute force to grow many trees with different parameters and compare their performance.

# The mean squared error is very low for this choice of parameters:
Y.hat.tuned <- predict(r.forest)
mse(Y.hat.tuned[, "predictions"] - Y) 
# For comparison see the graph of MSE against tree number

# It is also possible to tune only selected parameters, while defining others:
r.forest <- regression_forest(X, Y, sample.fraction = 0.4, tune.parameters = c("num.trees", "mtry"))
Y.hat <- predict(r.forest)
mse(Y.hat[, "predictions"] - Y)
# The MSE is much higher in this case, compared to tuning all parameters.

################################################################################
# Causal inference using forests

# Now we will take a look at the causal_forest() function. Keep in mind that all
# parameters and features discussed above for regression_forest() work the same way
# in causal_forest.

# Separate W from the covariate matrix again:
X <- X[, 1:10]
# Remember: Y is a vector containing the outcome variable, X is a matrix containing
# the covariates, and W is a binary vector indicating which observations received
# treatment.

# Additionally, define treatment propensity W.hat, which was assigned to be 0.5 when
# creating the data:
W.hat <- 0.5

# Get random forest predictions of y:
Y.predictions <- Y.hat.tuned[, "predictions"]

# We can now train a causal forest:
causal_forest(X, Y, W, Y.predictions, W.hat)

# Note that only X, Y, and W are required inputs, as the function will automatically
# use random forests to predict W.hat and Y.predictions on its own if no arguments
# are passed:
causal_forest(X, Y, W)

c.forest <- causal_forest(X, Y, W, W.hat)
# The predict funtion for causal_forest objects will return treatment effect estimations
# for every observation: In other words, the function returns an estimate of how
# big the treatment effect would have been for any given observation, if it was observed
# in both the treated and untreated state:
Y.hat <- predict(c.forest)
head(Y.hat)

# A simple estimator of the  average treatment effect can now be retrieved manually:
mean(Y.hat[, "predictions"])

# Or more robustly via an inbuilt function:
ate <- average_treatment_effect(c.forest)
ate
# We can also use this function to estimate the Average treatment effect on the
# treated (ATT) or the Average treatment effect on the untreated:
average_treatment_effect(c.forest, target.sample = ("treated"))
average_treatment_effect(c.forest, target.sample = ("control"))

# As discussed earlier, all input parameters from regression_forest() transfer to
# causal_forest:
causal_forest(X, Y, W, W.hat, num.trees = 100, sample.fraction = 0.5, mtry = min(ceiling(sqrt(ncol(X)) + 20)))

# But how do we know the underlying assumptions hold? Of course we can not proof
# they hold true but for some it is possible to gather indicative evidence.
# 1. Show that covariates are balanced across the treated and untreated:

# Inverse propensity scores:
IPW <- ifelse(W == 1, 1 / W.hat, 1 / (1 - W.hat))
# Name X columns:
colnames(X) <- make.names(1:p)
# Create a dataframe containing the plot data:
plot.df <- data.frame(
  value = as.vector(X),
  variable = colnames(X)[rep(1:p, each = n)],
  W = as.factor(W),
  IPW = IPW
)

# Plot the distribution of each covariate in treatment and control group:
ggplot(plot.df, aes(x = value, weight = IPW, fill = W)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 30) +
  facet_wrap(~variable, ncol = 2)
# The covariate distribution in the treated and untreated groups does look very 
# similar. The Covariates appear to be balanced across the treated and untreated.

# 2: Treatment assignment/ valid control group.
# Predict W on X using a random forest:
w.forest <- regression_forest(X, W)
propensity.hat <- predict(w.forest)
hist(propensity.hat[, "predictions"], main = "Estimate of propensity score distribution",
     xlab = "Propensity score")
# We show that the propensity score estimates appear to be normally distributed around 0.5.
# Thus, treatment assignment is not predictable based on the covariates and appears
# to be random. We know this to be true, since W was set to 0.5 when creating the data.
# Note however, that random assignment is not a necessary assumption: The estimate
# should be unbiased as long as there is a positive probability of being treated
# or untreated for every observation: W should not be close to 0 or 1. If treatment
# was guaranteed or ruled out for some observations, treatment effect estimation
# would of course be impossible since there simply are no similar observations in
# the other group.

# Clear the environment:
rm(list = ls())

################################################################################
# SECTION 2: Heterogeneity analysis:
# I will now turn to analysing heterogeneity in the Carhalho et al. (2019) dataset.
# The RCT tries to measure the impact of poverty cognitive ability. Specifially,
# participants are randomly assigned to partake in an online survey either shortly
# before or  after they received their paycheck.
# The main points of orientation in the analysis are the grf documentation(2023)
# , the grf github repository (2023), and Farbmacher et al. (2021).

# In a first step get an overview of the data and prepare it for usage in a random
# forest.

# Load the data:
data <- read.csv("carvalho2016.csv")

# Get an overview:
colnames(data)
summary(data)
# There are 3 measures of the outcome, a treatment indicator and 24 covariates. 

# Partition the data into outcome, covariates, and treatment indicator:
# A vector Y containing the main outcome variable:
Y <- data$outcome.num.correct.ans

# The binary vector W containing a treatment indicator for each observation:
W <- data$treatment

# Lastly the matrix of covariates X.
# We exclude the first 4 columns since there are 3 measures of Y as well as W:
X <- data[-(1:4)]

# In a first step, check whether treatment assignment is predictable based on the
# covariates as shown before. Since the data stems from an RCT, treatment should
# be randomly assigned and therefore W.hat should be distributed around 0.5:
propensity.forest <- regression_forest(X, W)

# Predict probability of treatment assignment for each value:
propensity.hat <- predict(propensity.forest)
hist(propensity.hat[, "predictions"], main = "Estimate of propensity score distribution", xlab = "Propensity score")
# Indeed, the propensity appears to be distributed normally around 0.5.

# But what if it was less clear? What if propensity was correlated to an x in the
# covariate space and we want to understand the relation?
# We could also investigate this with regard to an individual covariate - for instance
# age:

# Load the columns we need into a separate data frame:
plotdata <- data.frame(propensity.hat = propensity.hat[, "predictions"], X$age)

# Get the average propensity.hat value for each age:
average_phat <- plotdata %>%
  group_by(X.age) %>%
  summarize(average_phat = mean(propensity.hat))

# Plot propensity.hat on the y and age on the x axis. The average propensity.hat for a specific
# age is added as a red line.
ggplot(data = plotdata, aes(x = X.age, y = propensity.hat)) +
  geom_point(size = 0.85) +
  geom_line(data = average_phat, aes(y = average_phat), color = "red") +
  coord_cartesian(ylim = c(0.8 * min(plotdata$propensity.hat), max(plotdata$propensity.hat) * 1.2)) +
  labs(x = "Age", y = "Propensity", title = "Propensity as a function of age") +
  theme(plot.title = element_text(hjust = 0.5))
# The propensity does not appear to vary significantltly across age.

# Clean up the environment:
rm(plotdata, average_phat, propensity.forest, propensity.hat)

# Before starting to estimate treatment effects one additional step is required to
# prepare the data in this case: The dataset only contains ~2500 observations. In
# order to estimate heterogeneous treatment effects with reasonable error rates the
# number of covariates in relation to the number of observations needs to be reduced.
# But how do we decide which covariates to keep and which to drop?

# Step 1: Predict Y on X:
Y.forest <- regression_forest(X, Y)
Y.hat <- predict(Y.forest)$predictions
# Step 2: Get variable importance measure:
var_imp.Y <- variable_importance(Y.forest)
# Step 3: Order by importance:
var_imp.Y <- order(var_imp.Y, decreasing = TRUE)
# Get the names of the 10 most important columns:
columns_to_keep <- colnames(X)[var_imp.Y[1:10]]
# Discard all other columns in X:
X.reduced <- X[, columns_to_keep]
# X.reduced now contains the 10 most relavant values in the random forest.

################################################################################
# SECTION 2.1: Best linear projection:
# Now that there are only 10 columns left move to analysing treatment effects:

# We know W = 0.5, because the data stems form an RCT and because of the propensity
# estimation.
W.hat <- 0.5
# Estimate a causal forest:
c.forest <- causal_forest(X.reduced, Y, W, W.hat = W.hat)
# Calculate the ATE:
average_treatment_effect(c.forest)
# There does not appear to be a treatment effect on average.
# But maybe the treatment effect varies between observations with different covariates?


# The function best_linear_projection returns estimates of the linear relation between
# the (conditional) ATE and the covariates. While we have no reason to assume that
# the relation is linear, it presents a good first step to investigate if heterogeneity
# is present and which variables appear to drive it.
best_linear_projection(c.forest, X.reduced)
# While none of the relations appears to be significant at first glance, heterogeneity
# does appear to be present to some degree.

################################################################################
# SECTION 2.2: Recreate  graph from Carter et al. (2019):


# Define a function to create the graph:
# Parameters in order: A trained causal forest, the dataframe to predict, the column
# that contains actual values, the percentile at which covariates are held constant
# and the graph label on the x axis.
# Note: The steps are explained in detail on one example below.
percentile_graph <- function(c.forest, X.percentile, col, percentile, x_label) {
  predict.percentile <- predict(c.forest, X.percentile, estimate.variance = TRUE)
  X.percentile$prediction <- predict.percentile$predictions
  X.percentile$prediction.variance <- predict.percentile$variance.estimates
  plot.percentile <- ggplot(X.percentile, aes(x = X.percentile[[col]], y = prediction)) +
    geom_line() +
    geom_hline(yintercept = 0, color = "red") +
    geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
    geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
    labs(
      title = paste("Covariates at percentile", percentile), y = "Predicted treatment effect",
      x = x_label
    ) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))
  plot.percentile
}

# In order to create the plots we need to know the values of all covariates at their
# respective 25th, 50th, and 75th percentiles. We will store this information in
# dataframes for each percentile. Subsequently, one covariate/column at a time is
# replaced with actual values for a specific covariate. 
# Note that the impacts of moving the covariates to the xth percentile  on the expected
# treatment effect can be heterogenous itself.
# Interpretation of these graphs is thus not always straight forward. It could be
# that moving covariate A to the 25th percentile increases expected treatment effects
# while moving covariate B to the 25th percentile reduces them relative to the ATE.
# In order to simplify interpretation I will create the graphs only based on the
# variables "age", "household.income", "education", and "current.income": Arguably,
# an individual in the lower percentiles of these variables could be seen as disadvantaged.
# Therefore one could expect a stronger reduction in test scores, which is a testable
# theory and allows to interpret the graphs more clearly.
X.graphs <- X.reduced[c("age", "household.income", "education", "current.income")]
c.forest <- causal_forest(X.graphs, Y, W, W.hat = W.hat)

# Create percentile dataframes:
length <- length(X.graphs$age) - 1
X.temp <- data.frame(matrix(ncol = ncol(X.graphs), nrow = 0))

# Create a dataframe of the same dimension of X.graphs, but only containing the
# value at the 25th percentile for each variable:
X.25 <- apply(X.graphs, 2, quantile, probs = 0.25)
X.25 <- rbind(X.temp, X.25)
colnames(X.25) <- colnames(X.graphs)
X.25 <- rbind(X.25, X.25[rep(1, length), ])

# Identical appproach for the median/ 50th percentile:
X.50 <- apply(X.graphs, 2, quantile, probs = 0.5)
X.50 <- rbind(X.temp, X.50)
colnames(X.50) <- colnames(X.graphs)
X.50 <- rbind(X.50, X.50[rep(1, length), ])

# And the 75th percentile:
X.75 <- apply(X.graphs, 2, quantile, probs = 0.75)
X.75 <- rbind(X.temp, X.75)
colnames(X.75) <- colnames(X.graphs)
X.75 <- rbind(X.75, X.75[rep(1, length), ])

# Clean up environment:
rm("X.temp", "length")

# Create new dataframes which contain actual values for the age variable, but are
# held constant at the percentile values for all other covariates:

# Copy the percentile dataframes:
X.25.age <- X.25
X.50.age <- X.50
X.75.age <- X.75

# Replace the age percentile values with actual observations:
X.25.age$age <- X.graphs$age
X.50.age$age <- X.graphs$age
X.75.age$age <- X.graphs$age

# Predict conditional treatment effects for the X.25.age dataframe:
predict.25.age <- predict(c.forest, X.25.age, estimate.variance = TRUE)

# Load the relevant data into the X.25.age dataframe:
X.25.age$prediction <- predict.25.age$predictions
X.25.age$prediction.variance <- predict.25.age$variance.estimates

# Create a plot with age on the X axis and the predicted treatment effect at the specific
# covariate combination on the y axis. Standard errors are added as dotted lines
# around the estimates and 0 (no treatment effect) is marked as a red line for reference.
plot.age.25 <- ggplot(X.25.age, aes(x = age, y = prediction)) +
  geom_line() +
  geom_hline(yintercept = 0, color = "red") +
  geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
  geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
  labs(
    title = "Covariates at percentile 25", y = "Predicted treatment effect",
    x = "Age"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))
plot.age.25

# Create the graph for age with covariates at the 50th percentile:
predict.50.age <- predict(c.forest, X.50.age, estimate.variance = TRUE)
X.50.age$prediction <- predict.50.age$predictions
X.50.age$prediction.variance <- predict.50.age$variance.estimates
plot.age.50 <- ggplot(X.50.age, aes(x = age, y = prediction)) +
  geom_line() +
  geom_hline(yintercept = 0, color = "red") +
  geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
  geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
  labs(
    title = "Covariates at percentile 50", y = "Predicted treatment effect",
    x = "Age"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))

# Create the graph for age with covariates at the 75th percentile:
predict.75.age <- predict(c.forest, X.75.age, estimate.variance = TRUE)
X.75.age$prediction <- predict.75.age$predictions
X.75.age$prediction.variance <- predict.75.age$variance.estimates
plot.age.75 <- ggplot(X.75.age, aes(x = age, y = prediction)) +
  geom_line() +
  geom_hline(yintercept = 0, color = "red") +
  geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
  geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
  labs(
    title = "Covariates at percentile 75", y = "Predicted treatment effect",
    x = "Age"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))

# The procedure of creating the graph should now be clear. For the rest of the 
# covariates I will use the percentile_graph function, which was defined at the
# very beginning of section 2.

# Create the dataframes containing household.income values while all other
# columns are held constant at the respective percentiles:
X.25.household.income <- X.25
X.50.household.income <- X.50
X.75.household.income <- X.75

# Replace the household income column with actual observations:
X.25.household.income$household.income <- X.graphs$household.income
X.50.household.income$household.income <- X.graphs$household.income
X.75.household.income$household.income <- X.graphs$household.income

# Create the graph for household income with covariates at the different percentiles:
plot.household.income.25 <- percentile_graph(c.forest, X.25.household.income, "household.income",
                                             "25", "Houesehold Income")
plot.household.income.50 <- percentile_graph(c.forest, X.50.household.income, "household.income",
                                             "50", "Houesehold Income")
plot.household.income.75 <- percentile_graph(c.forest, X.75.household.income, "household.income",
                                             "75", "Houesehold Income")

# Create the dataframes containing education values while all other
# columns are held constant at the respective percentiles:
X.25.education <- X.25
X.50.education <- X.50
X.75.education <- X.75

# Replace the education columns with actual values:
X.25.education$education <- X.graphs$education
X.50.education$education <- X.graphs$education
X.75.education$education <- X.graphs$education

# Create the graph for education with covariates at the different percentiles: 
plot.education.25 <- percentile_graph(c.forest, X.25.education, "education", "25", "Education")
plot.education.50 <- percentile_graph(c.forest, X.50.education, "education", "50", "Education")
plot.education.75 <- percentile_graph(c.forest, X.75.education, "education", "75", "Education")

# Create the dataframes containing current income values while all other
# columns are held constant at the respective percentiles:
X.25.current.income <- X.25
X.50.current.income <- X.50
X.75.current.income <- X.75

# Replace the current income colum with actual values:
X.25.current.income$current.income <- X.graphs$current.income
X.50.current.income$current.income <- X.graphs$current.income
X.75.current.income$current.income <- X.graphs$current.income

# Remove outliers that impedes the graph otherwise:
X.25.current.income <- X.25.current.income[X.25.current.income$current.income < 6000, ]
X.50.current.income <- X.50.current.income[X.50.current.income$current.income < 6000, ]
X.75.current.income <- X.75.current.income[X.75.current.income$current.income < 6000, ]

# Create the graph for current income with covariates at the different percentiles:
plot.current.income.25 <- percentile_graph(c.forest, X.25.current.income, "current.income", "25", "Current Income")
plot.current.income.50 <- percentile_graph(c.forest, X.50.current.income, "current.income", "50", "Current Income")
plot.current.income.75 <- percentile_graph(c.forest, X.75.current.income, "current.income", "75", "Current Income")

# Create a big summary plot with all graphs:

# Three columns with 5cm width each:
widths <- unit(c(8, 8, 8), c("cm", "cm", "cm"))
# Four rows with 3cm height each:
heights <- unit(c(6, 6, 6, 6), c("cm", "cm", "cm"))
summary.plot <- grid.arrange(plot.age.25, plot.age.50, plot.age.75, plot.household.income.25, plot.household.income.50,
  plot.household.income.75, plot.education.25, plot.education.50,
  plot.education.75, plot.current.income.25, plot.current.income.50,
  plot.current.income.75,
  nrow = 4, ncol = 3, widths = widths, heights = heights
)

# Save the plot as a png file as the main output:
ggsave("summary_plot.png", summary.plot, width = 24, height = 24, units = "cm")

# Right now the no treatment effect line is marked in red. However, the estimated 
# treatment effect at the point where all covariables are held constant seems like 
# a more sensible comparative value. Let's add it to the graph:

# In order to archive this, we first need an estimate of the treatment effects
# at the 25th, 50th, and 75th Covariate percentile.

predict.25 <- predict(c.forest, X.25, estimate.variance = FALSE)
predict.50 <- predict(c.forest, X.50, estimate.variance = FALSE)
predict.75 <- predict(c.forest, X.75, estimate.variance = FALSE)

# Now we merge the values of the predicted treatment effect at their percentiles to the dataframes:
X.25.age$percentile_cate.25 <- predict.25$predictions
X.50.age$percentile_cate.50 <- predict.50$predictions
X.75.age$percentile_cate.75 <- predict.75$predictions

# Also add the variable percentiles:
percentiles <- quantile(X.50.age[, "age"], probs = c(0.25, 0.5, 0.75))

# And now we can include them in the plots:
ggplot(X.25.age, aes(x = age, y = prediction)) +
  geom_line() +
  geom_hline(yintercept = 0, color = "red") +
  geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
  geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
  geom_hline(yintercept = X.25.age$percentile_cate, color = "blue", linetype = "dashed") +
  geom_vline(xintercept = percentiles[1], linetype = "twodash", color = "magenta", linewidth = 0.4) +
  labs(
    title = "Covariates at percentile 25", y = "Predicted treatment effect",
    x = "Age"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))

ggplot(X.50.age, aes(x = age, y = prediction)) +
  geom_line() +
  geom_hline(yintercept = 0, color = "red") +
  geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
  geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
  geom_hline(yintercept = X.50.age$percentile_cate, color = "blue", linetype = "dashed") +
  geom_vline(xintercept = percentiles[2], linetype = "twodash", color = "magenta", linewidth = 0.4) +
  labs(
    title = "Covariates at percentile 50", y = "Predicted treatment effect",
    x = "Age"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))

ggplot(X.75.age, aes(x = age, y = prediction)) +
  geom_line() +
  geom_hline(yintercept = 0, color = "red") +
  geom_line(aes(y = prediction - prediction.variance), linetype = "dotted") +
  geom_line(aes(y = prediction + prediction.variance), linetype = "dotted") +
  geom_hline(yintercept = X.75.age$percentile_cate, color = "blue", linetype = "dashed") +
  geom_vline(xintercept = percentiles[3], linetype = "twodash", color = "magenta", linewidth = 0.4) +
  labs(
    title = "Covariates at percentile 75", y = "Predicted treatment effect",
    x = "Age"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 10))

# Note how the treatment effect conditional on age (black) always crosses the treatment
# effect at the percentile at which the other covariates are held constant  (blue).
# This is marked with a pink horizontal line in all three graphs, respectively presenting
# the 25th, 50th, and 75th percentile of the age variable. At those locations, all
# covariates are at the same percentile and therefore the estimated treatment effects
# will be equal.


# Clear the environment, of all but the listed variables:
rm(list = setdiff(ls(), c("X.reduced", "Y", "W", "X")))


################################################################################
# SECTION 2.3: Ranked weighted average treatment effect:
# The package offers an alternative way of investigating heterogeneity: Rank-weighted
# average treatment effects (RATE), which are used to evaluate targeting rules, geared
# towards identifying units likely to experience particularly high/low treatment effects.

# The code will be structured as follows:
# We will use half of the data to generate predictions of the conditional average
# treatment effects (CATE). We will then train a second causal forest for
# evaluation on the other half of the data. This evaluation forest as well as the
# predictions generated by the first forest are then used to calculate the ranked
# weighted average treatment effect (RATE). I will explain the intuition behind
# this metric below in the code on an example.

Y.forest <- regression_forest(X, Y)
Y.hat <- predict(Y.forest)$predictions

# In order to estimate ranked weighed average treatment effects we do need to partition
# the data into a training and evaluation set, as results would be biased otherwise:
train <- 1:(nrow(X.reduced) / 2)
W.hat <- 0.5

# Train the first causal forest and use it to predict conditional treatment effects
# for the half of the data it was not trained on:
train.forest <- causal_forest(X.reduced[train, ], Y[train], W[train], Y.hat = Y.hat[train], W.hat = W.hat)
tau.hat.eval <- predict(train.forest, X.reduced[-train, ])$predictions

# Train the second causal forest for evaluation:
eval.forest <- causal_forest(X.reduced[-train, ], Y[-train], W[-train], Y.hat = Y.hat[-train], W.hat = W.hat)

# The rank_average_treatment_effect function calculates the RATE.
# The evaluation forest (first input) is used to calculate the RATE based on the
# conditional average treatment effects calculated with the train forest.
# I will explain in more detail based on the graphs below.
rate.cate <- rank_average_treatment_effect(eval.forest, list(cate = -1 * tau.hat.eval))
rate.age <- rank_average_treatment_effect(eval.forest, list(age = X[-train, "age"]))

# The curve in the plot below is referred to as the targeting operator characteristic.
# It compares the benefit of treating only a fraction of the population relative to
# the average treatment effect.
# The units must be ranked by some rule and are divided into quantiles based on
# this ranking. In the first plot the ranking is based on the CATEs with the lowest
# estimated treatment effects being in the lower quantiles (keep in mind that negative
# treatment effects are expected):
plot(rate.cate, ylab = "Number of correct answers", main = "TOC: By most negative CATEs")

# However, the ranking could also be based on a single covariate - for instance by
# decreasing age:
plot(rate.age, ylab = "Number of correct answers", main = "TOC: By decreasing age")

# The RATE is then computed as a weghted average of the TOC: Think of it as the 
# integral over the TOC curve but below 0.
# A particularly high (or as in this case low) value would suggest that the proposes
# targeting rule (CATE or age) does a good job at identifying units with high treatment
# effects:
rate.cate
rate.age
# For an in detail explanation and derivation of the RATE see Yadlowsky et al. (2021).

# Remove all variables but the ones listed:
rm(list = setdiff(ls(), c("X.reduced", "Y", "W", "W.hat", "X")))

################################################################################
# SECTION 2.4: Policytree
# The sister package policytree (developed by Sverdrup et al. (2021)) offers a more
# direct way to derive targeting rules: It allows to grow decision trees specifically
# to partition the population into groups with higher/ lower predicted treatment 
# effects.

# An interesting feature is that the dataset used to grow the policy tree does not
# have to contain the same covariates as the one used to grow the causal forest.
# This seems useful, because in a real world scenario a subset of the covariates
# could be much easier to acquire for a greater sample:
c.forest <- causal_forest(X.reduced, Y, W, W.hat = W.hat)
# Define subset of the data to grow the policy tree:
X.policy <- X.reduced[c("age", "household.income", "education", "current.income")]

# DR.scores contains a point estimate for each observation if it was treated/
# was in the control group:
dr.scores <- double_robust_scores(c.forest)
head(dr.scores)

# Grow the policy tree based on the data subset and dr.scores. The depth parameter
# defines how "deep" the tree should grow and thereby how many groups the sample
# is split into. Feel free to increase the value but note that depths of 3 or higher
# can take very long to run:
pol_tree <- policy_tree(X.policy, dr.scores, depth = 2, min.node.size = 100)
plot(pol_tree, c("intervention", "no intervention"))

# Right now the tree is "giving advice" simply based on the CATE within each leaf
# being positive or negative.
# But what if the policymaker for instance is only willing to intervene in groups
# for which the treatment effect is strongly negative due to costs of treatment?
# In order to create a policy tree that is useful in such a scenario we can introduce
# a penalty term for the treated:
penalty <- 1.5
dr.rewards <- dr.scores
# We add the penalty because the expected treatment effect is negative:
dr.rewards[, "treated"] <- dr.rewards[, "treated"] + penalty
# Grow the new tree:
tree <- policy_tree(X.policy, dr.rewards, depth = 2, min.node.size = 100)
plot(tree, leaf.labels = c("intervenention", "no intervention"))
# Predict which leaf each observation would fall into:
node.id <- predict(tree, X.policy, type = "node.id")
# Calculate the average point score within each leaf:
values <- aggregate(dr.rewards,
  by = list(leaf.node = node.id),
  FUN = function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x)))
)
# Calculate the "within-leaf" treatment effect:
values[, "Within leaf ATE"] <- values[, "treated"][, "mean"] - values[, "control"][, "mean"]
print(values)



# References in scientific literature:
# Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
# Carter, M. R., Tjernström, E., & Toledo, P. (2019). Heterogeneous impact dynamics of a rural business development program in Nicaragua. Journal of Development Economics, 138, 77-98.
# Carvalho, L. S., Meier, S., & Wang, S. W. (2016). Poverty and economic decision-making: Evidence from changes in financial resources at payday. American economic review, 106(2), 260-284.
# Farbmacher, H., Kögel, H., & Spindler, M. (2021). Heterogeneous effects of poverty on attention. Labour Economics, 71, 102028.
# Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., & Wager, S. (2020). policytree: Policy learning via doubly robust empirical welfare maximization over trees. Journal of Open Source Software, 5(50), 2232.
# Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2021). Evaluating treatment prioritization rules via rank-weighted average treatment effects. arXiv preprint arXiv:2111.07966.

# Other references:
# GRF package documentation (2023): https://grf-labs.github.io/grf/index.html
# GRF Github repository (2023): https://github.com/grf-labs/grf
# Policytree documentation (2022): https://grf-labs.github.io/policytree/articles/policytree.html
