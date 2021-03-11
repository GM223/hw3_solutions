
# HW3: Intro to Nonlinear Trajectory Optimization 
## Due date: Wednesday, March 24 

In this homework you will implement some foundational methods for solving simple nonlinear trajectory optimization problems. Here's an overview of the problems:
1. Implement iLQR to get a planar quadrotor to do a simple flip. Track with TVLQR to make it robust to wind.
2. Write a sequential-quadratic-programming (SQP) solver to find a solution to the canonical cartpole problem. Generate a TVLQR controller to make it robust to model mismatch.

## Running the Autograder
The autograder setup has changed slightly for this assignment. We'll no longer use GitHub actions to run the autograder (since it didn't really work last time, maybe in future assignements...). You will run all your tests locally. To run the full test suite the same way we will when grading your submission follow these instructions:

1. Open a terminal in the root directory of your repo
2. Launch a Julia REPL
3. Enter the package manager using `]` and enter `activate .`
4. Launch the testing suite using `test hw3`

Each notebook now includes a `run_tests()` function at the end, that will run the test suite in your notebook. You can call that test at any point. It will just run the `q1.jl`, or
`q2.jl` files in the `test/` directory. Question 2 also has some additional test suites included in the middle of the notebook to help debug your methods.

## Submitting your homework
Make sure your repo lives under the Class Organization. This will be done automatically when you use the GitHub Classrooms link we send provide.

1. Add your info to the `studentinfo` function in [src/hw3.jl](https://github.com/Optimal-Control-16-745/hw3/blob/main/src/hw3.jl)
2. Create a release. Follow [these instructions](https://github.com/Optimal-Control-16-745/JuliaIntro/blob/main/docs/Submission%20Instructions.md) for more info on creating the release.

## Adding the Upstream Repo
We may release changes to the homework periodically if errors or bugs are found. Follow these instructions for linking your repo to the original template and pulling changes. It's always a good idea to branch your code before pulling from the upstream repo in case something goes wrong or the merge is particularly nasty. Do the right away after creating your repo. 
```
git remote add upstream https://github.com/Optimal-Control-16-745/hw3
git pull upstream main --allow-unrelated-histories
```
