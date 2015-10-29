% All-Pairs Shortest Paths
% David Bindel
% 2015-10-19
Team 9
--------
Batu (bi49)
Wensi (ww382)
Kenneth (kl545)

# Introduction

## The Floyd-Warshall Algorithm

The [Floyd-Warshall algorithm][fw-wiki] for computing all pairwise shortest
path lenghs in a graph has a computational pattern much like the
one for Gaussian elimination. There is a closely related algorithm
which is slightly more expensive – $O(n^3 \log n)$ time in general
rather than the $O(n^3)$ time required by Floyd-Warshall – but
which looks very much like matrix multiplication. In this
assignment, you will analyze the performance of a reference OpenMP
implementation of this method, and then implement and analyze your
own version using MPI.

As usual, you are allowed to use any references that you find, with
appropriate citations.  I know that people have worked on fast
Floyd-Warshall on GPUs; you may also find prior work from when I
taught the class in 2011 and used this assignment!

[fw-wiki]: https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm

## Your mission

You are provided with a reference OpenMP implementation (`path.c`).
For this assignment, you should attempt three tasks:

1.  *Profiling*:  The current code is not particularly tuned, and there
    are surely some bottlenecks.  Profile the computation and
    determine what parts of the code are slowest.  I encourage you to
    use profiling tools (e.g. VTune Amplifier), but you may also
    manually instrument the code with timers.

2.  *Parallelization*: The current code is parallelized with OpenMP.
    You should also parallelize your code using MPI, and study the
    speedup versus number of processors on both the main cores on the
    nodes and on the Xeon Phi boards.  Set up both strong and weak
    scaling studies, varying the number of threads/processes you employ.

3.  *Tuning*:  You should tune your code in order to get it to run as
    fast as possible.  For tuning, you may focus on either the OpenMP
    or the MPI version of the code.  The computational pattern is much
    like that of parallel Gaussian elimination, and in addition to
    tuning the parallelism, I encourage you to use the tools you
    learned about in matrix multiply (vectorization, blocking).

The primary deliverable for your project is a report that describes
your performance experiments and attempts at tuning, along with what
you learned about things that did or did not work.  Good things for
the report include:

- Profiling results
- Speedup plots and scaled speedup plots
- Performance models that predict speedup

In addition, you should also provide the code, and ideally scripts
that make it simple to reproduce any performance experiments you've
run.

## Logistical notes

### Timeline

As with the previous assignment, this assignment involves two stages.
By Nov 3, you should submit your initial report (and code) for peer
review; reviews are due by Nov 5.  Final reports are due one week
later (Nov 12).  I hope this project is more straightforward than the
shallow water equation, so that many of you will be able to wrap up
early.

### Peer review logistics

Since the first assignment, GitHub has added a feature to
[attach PDF files to issues and pull request comments][pdf].  You
should take advantage of this feature to submit your review as a
comment on the pull request for the group you are reviewing.
You should still look at the codes from the other groups, though!

[pdf]: https://github.com/blog/2061-attach-files-to-comments

### Notes on MPI on the Phi boards

I have succeeded in running MPI jobs on the Phi boards, but it seems
to take quite a while for jobs to start.  We have also had some
difficulties getting authentication working properly, and it's
possible that you will run into hiccups.  Please give it a try, but if
you start running into trouble with MPI on the Phi, ask questions
early and often -- on Piazza, so we can all figure it out together!
