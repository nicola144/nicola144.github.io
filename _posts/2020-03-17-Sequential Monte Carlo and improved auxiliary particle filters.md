---
title : "Sequential Monte Carlo and Improved Auxiliary Particle Filters"
---
<head>
<style> 
#example1 {
  border: 1px solid;
  padding: 5px;
  box-shadow: 5px 10px;
}
</style>
</head>

In this post, my aims are: 
* Introduce Bayesian inference in state space models
* Introduce approximate inference using importance sampling, in state space models. I will try to present and compare different ways of deriving the algorithms that I found in the literature, trying to unify them in a way that I have not seen explicitly elsewhere.
* Finally, describe the Auxiliary Particle Filter, its diverse intepretations and the recent Improved Auxiliary Particle Filter by Elvira et al [1]. I will illustrate the IAPF by reproducing the results of Elvira et al. [3]. 

The somewhat comprehensive "tutorial" and introduction to the topic arose from my feeling that the only content online that covers particle filters is either very theoretical material , or programming-oriented tutorials that completely lack motivation for how and why the algorithms are constructed as they are.


1. [Brief introduction to sequential inference](#introduction)
    1. [General Bayesian Filtering](#generalfilter)
    2. [Recursive Formulations](#recursive)
2. [Particle Filtering](#pf)
    1. [Basics of Monte Carlo and Importance Sampling](#basics)
    2. [Choice of proposal and variance of importance weights](#isproposal)
    3. [Sequential Importance Sampling](#sis)
    4. [Resampling](#resampling)
3. [Propagating particles by incoporating the current measurement](#apf)
    1. [The effect of using the locally optimal proposal](#optimalproposal)
    2. [The Auxiliary Particle Filter](#apf2)
        1. [A first intepretation: a standard SMC algorithm with a different $$\gamma$$](#firstapf)
        2. [The original intepretation of APF and Marginal Particle Filters](#marginalpf)
4. [The Multiple Importance Sampling Interpretation of PF](#mis)
    1. [The Improved Auxiliary Particle Filter](#iapf)


## Brief introduction to sequential inference <a name="introduction"></a>

In Bayesian inference we want to update our beliefs on the state of some random variables, which could represent parameters of a parametric statistical model or represent some unobserved data generating process. Focussing on the "updating" perspective, the step to using Bayesian methods to represent dynamical systems is quite natural. The field of statistical signal processing has been using the rules of probabilities to model object tracking, navigation and even.. spread of infectious diseases. 
The probabilistic evolution of a dynamical system is often called a *state space model*. This is just an abstraction of how we think the state of the system evolves over time. Imagine we are tracking a robot's position (x,y coordinates) and bearing: these constitute a three dimensional vector. At some specific timestep, we can have a belief, i.e. a probability distribution that represents how likely we think the robot is currently assuming a certain bearing etc. If we start with a prior, and define some likelihood function/ sampling process that we believe generates what we observe, we can update our belief over the system's state with the rules of probabilty.
Let the (uknown) state of the system at time $t$ be the vector valued random variable $\mathbf{s}_{t}$.

We observe this state through a (noisy) measurement $\mathbf{v}_{t}$ (where v stands for visible). 

Now we have to start making more assumptions. What does our belief on $\mathbf{s}_{t}$ depend on ? 

Suprisingly to me, it turns out for **a lot** of applications it just needs to depend on the $$\mathbf{s}$$tate at the previous timestep. 
In other words, we can say that $$\mathbf{s}_{t}$$ is sampled from some density $$f$$ conditional on $$\mathbf{s}_{t-1}$$: <br>

$$ 

\color{blue}{\text{Transition density}}: \qquad \mathbf{s}_{t} \sim \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) 

\tag{1}\label{eq1}
$$

Further, usually the observation or $$\mathbf{v}$$isible is sampled according to the current state:

$$
\color{green}{\text{Observation density}}: \mathbf{v}_{t} \sim \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) \tag{2}\label{eq2}
$$

It is reasonable to assume this: if we take a measurement, we don't expect its outcome to be dependent on previous states of the system, just the current one ($$\color{blue}{f}$$ and $$\color{green}{g}$$ seem arbitrary but they are common in the literature). For example, a classic Gaussian likelihood for would imply that the belief over $$\mathbf{v}_{t}$$ is a Normal , with the mean being a linear combination of the state's coordinates.

This collection of random variables and densities defines the state space model completely. It is worth, if you see this for the first time, reflecting on the particular assumptions we are making. How the belief on $\mathbf{s}$ evolves with time could depend on many previous states; the measurement could depend on previous measurements, if we had a sensor that degrades over time, etc... I am not great at giving practical examples, but if you are reading this, you should be able to see that this can be generalized in several ways. 
Note that a lot of the structure comes from assuming some variables are (conditionally) independent from others. The field of probabilistic graphical models is dedicated to representing statistical independencies in the form of graphs (nodes and edges). One benefit of the graphical representation is that it makes immediately clear how flexible we could be. I am showing the graphical model for the described state space model in Figure 1, below. 


![hhm]({{ '/assets/images/tikz2.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1: Graphical model for the typical, first order Markov state space. Shaded nodes represent random variables whose value has been observed. In this model, each observation or "visible" is generated by some unobserved state at each time step.*

In short, when the transition density and the observation densities are linear combination of their inputs with additive, i.i.d. Gaussian noise, then the state space model is often called a Linear Dynamical System (LDS). When variables are discrete, it is often called Hidden Markov Model (HMM). These are just labels. 

There are several tasks that we can perform on the state space model described above. Each of these has a fancy name, but one should of course recall that technically all we are doing is applying the sum and product rules. These tasks are associated to a *target* distribution which is the object of interest that we want to compute. Listing some of these: 

<p>
  <b>Filtering</b>: The target distributions are of the form: $p\left(\mathbf{s}_{t} | \mathbf{v}_{1: t}\right), \quad t=1, \ldots, T$. This represents what we have learnt about the system's state at time $t$, after observations up to $t$.<br>

  <b>Smoothing</b>:  The target distributions are of the form: $p\left(\mathbf{s}_{t} | \mathbf{v}_{1: T}\right), \quad t=1, \ldots, T$. This represent what we have learnt about the system's state after observing the *complete* sequence of measurements, and revised the previous beliefs obtained by filtering. <br>

<b>Parameter Estimation</b>: The target distributions are of the form: $p\left(\boldsymbol{\theta} | \mathbf{v}_{1: T}\right)= \int p\left(\mathbf{s}_{0: T}, \boldsymbol{\theta} | \mathbf{v}_{1: T}\right) \mathrm{d} \mathbf{s}_{0: T}$. The parameters $\boldsymbol{\theta}$ represent all the parameters of any parametric densities in the state space model. In the case that the transition and/or observation densities are parametric, and parameters are unknown, we can learn them from data by choosing those that both explain the observations well and also agree with our prior beliefs. Parameter estimation is sometimes referred to as <i>learning</i>, because parameters describe properties of sensors that can be estimated from data with machine learning methods. In other words, it is called learning just because it is cool. <br>
</p>



### General Bayesian Filtering <a name="generalfilter"></a>

#### Some notation/terminology
- As common in this field, I use the overloaded term of "distribution" to refer to densities, mass functions and distributions. Moreover the same notation is used for random variables and their realization ie. $$p(\mathbf{X} = \mathbf{x} \mid \mathbf{Z} = \mathbf{z}) = p(\mathbf{x} \mid \mathbf{z})$$ 
- The notation $$\mathbf{v}_{1:t}$$ means a collection of vectors $$ \left \{ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right \}$$
- Therefore, $$ p\left ( \mathbf{v}_{1:t} \right )$$ is a joint distribution: $$p\left ( \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right ) $$
- Integrating $$ \int p(\mathbf{x}_{1:t}) \mathrm{d}\mathbf{x}_{i:j}$$ means $$ \underbrace{\int \dots \int}_{j-i+1} p(\mathbf{x}_{1:t}) \mathrm{d}\mathbf{x}_{i} \mathrm{d}\mathbf{x}_{i+1} \dots \mathrm{d}\mathbf{x}_{j} $$
- The symbol $$:=$$ denotes a definition.

In this post, I am only concerned with filtering, and will always assume that any parameters of <span style="color:blue">transition</span> or <span style="color:green">observation</span> densities are known in advance. There are classes of algorithms that learn the parameters and perform inference at the same time, such as Particle Markov Chain Monte Carlo or SMC2. 

Let's start by deriving the filtering distribution in the state space model described without many assumption on the distributions.
Recall that the aim is to compute: $$ p\left(\mathbf{s}_{t} | \mathbf{v}_{1: t}\right)$$. Apply Bayes rule:  

$$
\require{cancel}
p\left(\mathbf{s}_{t} | \mathbf{v}_{1:t}\right) = \frac{ \overbrace{p \left( \mathbf{v}_{t} \mid \mathbf{s}_{t}, \cancel{\mathbf{v}_{1:t-1}} \right )}^{\mathbf{v}_t ~ \text{only dep. on} ~ \mathbf{s}_t} p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}_{1:t-1} \right )} = \frac{  \color{green}{g}\left( \mathbf{v}_{t} \mid \mathbf{s}_{t} \right ) p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}_{1:t-1} \right )} \tag{3}\label{eq3}
$$

If this equation is confusing, think of the previous measurements $$\mathbf{v}_{1:t-1}$$ as just a "context", that is always on the conditioning side, a required "input" to all densities involved, with Bayes rule being applied to $$\mathbf{s}_{t}$$ and $$\mathbf{v}_{t}$$.
We know the current measurements only depends on the state, therefore $$p \left( \mathbf{v}_{t} \mid \mathbf{s}_{t}, \mathbf{v}_{1:t-1} \right ) = p \left( \mathbf{v}_{t} \mid \mathbf{s}_{t} \right ) = \color{green}{g}( \mathbf{v}_{t} \mid \mathbf{s}_{t} )$$, and only the term on the right side of the numerator is left to compute. This term is a marginal of $$ \mathbf{s}_t$$, which means we have to integrate out anything else. If we were doing this very naively, each time we would integrate out all previous states, but by caching results a.k.a. Dynamic Programming, we only need to marginalize the previous state: 

$$
  p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right ) = \int p\left( \mathbf{s}_{t}, \mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1} \right ) \mathrm{d}\mathbf{s}_{t-1}
$$

Continuing, we split the joint with the product rule and exploit that the states are independent of previous measurements:

$$
\begin{equation}\begin{aligned}
  &= \int p\left( \mathbf{s}_{t} \mid  \mathbf{s}_{t-1}, \cancel{\mathbf{v}_{1:t-1}} \right ) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d}\mathbf{s}_{t-1} \\
  &= \int p\left( \mathbf{s}_{t} \mid  \mathbf{s}_{t-1} \right ) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d}\mathbf{s}_{t-1} \\
  &= \int \color{blue}{f}\left( \mathbf{s}_{t} \mid  \mathbf{s}_{t-1} \right ) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d}\mathbf{s}_{t-1}
\end{aligned}\end{equation}\tag{4}\label{eq4}$$


And we are done, if you notice that the right side term in the integral is the filtering distribution at $$t-1$$, which we have already computed recursively. 
In the literature names are given to the step that requires computing $$ p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right )$$ called *prediction*, because it's our belief on $$ \mathbf{s}_{t}$$ before observing the currrent measurement, and *correction* is the name given to the step $$ p\left(\mathbf{s}_{t} | \mathbf{v}_{1:t}\right) \propto \color{green}{g}\left( \mathbf{v}_{t} \mid \mathbf{s}_{t} \right ) \cdot p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right )$$, because we "correct" our prediction by taking into account the measurement. 

In a LDS all computations have closed form solutions, and this algorithm instantiates into the *Kalman Filter*. For discrete  valued random variables, if the dimensionalities are small we can also do exact computations and the label this time is *Forward-Backward* algorithm for HMMs. 
When variables are non-Gaussian and/or transition/observation densities are nonlinear function of their inputs, we have to perform approximate inference. 
By far the most popular method is to use Monte Carlo approximations, and more specifically importance sampling. When we use importance sampling to approximate the filtering distribution, this is called *particle filtering*. 

### Recursive formulations <a name="recursive"></a>

There is another way to derive generic computation steps to obtain the filtering distribution. It is common in the particle filtering literature to consider the sequential estimation of a different distribution to the filtering, namely: 

$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \propto p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t}) 
\end{aligned}\end{equation}\tag{5}\label{eq5}$$

Let's take its unnormalized version for simplicity. Applying Bayes' rule gives the following recursive relationship:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t}) = p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) 
\end{aligned}\end{equation}\tag{6}\label{eq6}$$

If you can't see why this holds, consider this simple example/subcase: 

$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{1}, \mathbf{s}_{2}, \mathbf{v}_{1}, \mathbf{v}_{2}) =   p(\mathbf{s}_{1}, \mathbf{v}_{1}) p(\mathbf{s}_2, \mathbf{v}_2 \mid \mathbf{s}_1, \mathbf{v}_1) = p(\mathbf{s}_{1}, \mathbf{v}_{1}) \color{blue}{f}(\mathbf{s}_{2} \mid \mathbf{s}_{1}) \color{green}{g}(\mathbf{v}_{2} \mid \mathbf{s}_{2})
\end{aligned}\end{equation}$$

Hopefully this convinces you that \eqref{eq6} is true. Then, let's return to the task of sequentially estimating $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$: 


$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) &= \frac{p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})}{p(\mathbf{v}_{1:t})} \\
&= \frac{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{1:t})} \\ 
&= \frac{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})}{p(\mathbf{v}_{1:t-1})} \frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} \\
&= p(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:t-1}) \frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})}
\end{aligned}\end{equation}\tag{7}\label{eq7}$$

Now that we've gone through all this, we are ready to show how to get the filtering distribution by simple marginalization of the expression we just found: 

$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t}) &= \int p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \mathrm{d} \mathbf{s}_{1:t-1} \\
&= \int p(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:t-1}) \frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} \mathrm{d} \mathbf{s}_{1:t-1}\\
&= \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} \int p(\mathbf{s}_{1:\color{red}{t-1}} \mid \mathbf{v}_{1:t-1}) \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \mathrm{d} \mathbf{s}_{1:t-1} \\
&= \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} \overbrace{\int p(\mathbf{s}_{1:\color{red}{t}} \mid \mathbf{v}_{1:t-1}) \mathrm{d} \mathbf{s}_{1:t-1}}^{= p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) ~ \text{by marginalization}} \\
&= \frac{p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} 
\end{aligned}\end{equation}\tag{8}\label{eq8}$$

Which is the indeed same result that we got through the prediction and correction steps. 

The two perspectives, namely the prediction-correction equations or the recursive formulations, can be both used to derive concrete algorithms in slightly different ways. Let's highlight the two most important equations for particle filtering: 

<div id="example1">
  
$$\begin{equation}\begin{aligned}
 p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) =  p(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:t-1}) \frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} 
\end{aligned}\end{equation}\tag{9}\label{eq9}$$
</div>
<br>

We can call this "Trajectory Filtering Distribution" (TFD), since it considers the sequential estimation of the whole trajectory of states. Similarly,

<div id="example1">
$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t}) = \frac{p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} 
\end{aligned}\end{equation}\tag{10}\label{eq10}$$
</div>
<br>

this can be called "State Filtering Distribution" (SFD).

## Particle filtering <a name="pf"></a> 

### Basics of Monte Carlo and Importance Sampling <a name="basics"></a> 
Most things aren't linear and/or Gaussian, so we need approximate inference. Specifically in the filtering/sequential Bayes literature, importance sampling based methods are more popular than deterministic approximations such as Laplace's method, Variational Bayes and Expectation Propagation. 
Recall that the Monte Carlo method is a general tool to approximate integrals, expectations, probabilities with random samples: 

$$
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} \approx \frac{1}{N} \sum_{n=1}^{N} f(\mathbf{x}_{n}) \qquad \mathbf{x}_n \sim p(\mathbf{x})
\tag{11}\label{eq11}$$

Where $$f(\mathbf{x})$$ is some generic function of $$\mathbf{x}$$. Monte Carlo approximations of this kind are very appealing since unbiased and consistent, and it is easy to show that the variance of the estimate is $$ \mathcal{O}(n^{-1})$$ *regardless* of the dimensionality of the vector $$\mathbf{x}$$. Another simple idea that we will use extensively in particle filtering is that these samples can not only be used to approximate integrals with respect to the target distribution $$p(\mathbf{x})$$, but also to approximate the target itself:

$$
p(\mathbf{x}) \approx \frac{1}{N}\sum_{n=1}^{N} \delta_{\mathbf{x}}(\mathbf{x}_{n})
$$

Where $$  \delta_{\mathbf{x}}(\mathbf{x}_{n})$$ is the Dirac delta mass evaluated at point $$\mathbf{x}_{n}$$. This is a function that is $$\infty$$ at its evaluation point and $$0$$ everywhere else, and satisfies $$ \int \delta_{\mathbf{x}} \mathrm{d}\mathbf{x} = 1$$, so that approximating the distribution itself recovers the previous result for expectations:

$$
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})]  \approx \int  f(\mathbf{x})  \frac{1}{N}\sum_{n=1}^{N} \delta_{\mathbf{x}}(\mathbf{x}_{n}) \mathrm{d}\mathbf{x} =   \frac{1}{N} \sum_{n=1}^{N} \int \delta_{\mathbf{x}}(\mathbf{x}_{n}) f(\mathbf{x}) \mathrm{d}\mathbf{x} = \frac{1}{N} \sum_{n=1}^{N} f(\mathbf{x}_n)
$$

This should be interpreted as an approximation to the underlying distribution and not of the density function. 
Often it is not possible to sample from the distribution of interest. Therefore we can use importance sampling, which is a technique based on the simple observation that we can sample from another, known distribution and assign a weight to the samples to represent their "importance" under the real target:

$$\begin{equation}\begin{aligned}
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] &= \int f(\mathbf{x}) \cdot p(\mathbf{x}) \mathrm{d}\mathbf{x} \\
&= \int \frac{f(\mathbf{x}) \cdot p(\mathbf{x})}{q(\mathbf{x})} \cdot q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \mathbb{E}_{q(\mathbf{x})} \left [ f(\mathbf{x}) \cdot \frac{p(\mathbf{x})}{q(\mathbf{x})} \right ] \\ 
&= \mathbb{E}_{q(\mathbf{x})} \left [ f(\mathbf{x}) \cdot w(\mathbf{x}) \right ]
\end{aligned}\end{equation}\tag{12}\label{eq12}$$

Under certain conditions, namely that $$ f(\mathbf{x}) \cdot p(\mathbf{x}) > 0 \Rightarrow q(\mathbf{x}) > 0$$, we have rewritten the expectation under a distribution of choice $$q(\mathbf{x})$$ called $$\color{#FF8000}{\text{proposal}}$$ which we know how to sample from. Note that it is not possible to have $$ q(\mathbf{x}) = 0$$, as we will never sample any $$\mathbf{x}_{i}$$ from $$q$$ such that this holds. The weight $$w(\mathbf{x})$$ can be intepreted as “adjusting” the estimate of the integral by taking into account that the samples were generated from the “wrong” distribution. Notice that Importance Sampling can be used to approximate also generic integrals which are not necessarily expectations.

Let's return in the context of Bayesian inference, where we have a target posterior distribution $$ \pi(\mathbf{x}) = p(\mathbf{x} \mid \mathcal{D}) $$ where $$ \mathcal{D}$$ is any observed data. 
In our state space model $$\mathcal{D} = \mathbf{v}_{1:t}$$, and $$\pi(\mathbf{x}) = p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$. Consider an integral of some function of $$ \mathbf{x}$$ under the posterior, often the ultimate object of interest: 

$$\begin{equation}\begin{aligned}
\mathcal{I} = \mathbb{E}_{\pi(\mathbf{x})}[f(\mathbf{x})] = \int f(\mathbf{x}) \pi(\mathbf{x}) 
\end{aligned}\end{equation}\tag{13}\label{eq13}$$

We can estimate this integral in two main ways with IS: the former which assumes that we know the normalizing constant of the posterior $$ \pi(\mathbf{x})$$, and the latter estimates the normalizing constant too by IS, with the same set of samples. Since in Bayesian inference we usually can only evaluate $$\pi(\mathbf{x})$$ up to a normalizing constant, let's examine the latter option, called *self-normalized* IS estimator:

$$\begin{equation}\begin{aligned}
\mathbb{E}_{\pi} \left [ f(\mathbf{x}) \right ] &= \int f(\mathbf{x}) \pi(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \int  f(\mathbf{x})\frac{\pi(\mathbf{x})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{p(\mathcal{D})q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \frac{1}{p(\mathcal{D})}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \frac{1}{\int p(\mathbf{x}, \mathcal{D}) \mathrm{d} \mathbf{x}}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \frac{1}{\int \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})}  q(\mathbf{x}) \mathrm{d} \mathbf{x}}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \frac{1}{\mathbb{E}_{q}\left [ \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} \right ]}
\cdot \mathbb{E}_{q}\left [ f(\mathbf{x}) \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} \right ] \\
&\approx \frac{1}{\cancel{\frac{1}{N}}\sum_{n=1}^{N} \frac{p(\mathbf{x}_n , \mathcal{D})}{q(\mathbf{x}_n)}}
\cdot ~ \cancel{\frac{1}{N}} \sum_{n=1}^{N} f(\mathbf{x}_n) \frac{p(\mathbf{x}_n, \mathcal{D})}{q(\mathbf{x}_n)} := \widehat{\mathcal{I}}_{SN}
\end{aligned}\end{equation}\tag{14}\label{eq14}$$

Where the ratio $$ \frac{p(\mathbf{x}_n, \mathcal{D})}{q(\mathbf{x}_n)} := \tilde{w}(\mathbf{x}_n)$$ plays the role of the (unnormalized) importance weight. The estimator $$\widehat{\mathcal{I}}_{SN}$$ can be shown to be biased. An important observation that is useful in particle filtering is that the normalizing constant estimate $$ Z \approx  \widehat{Z} = \frac{1}{N} \sum_{n=1}^{N} \frac{p(\mathbf{x}_n , \mathcal{D})}{q(\mathbf{x}_n)} = \frac{1}{N} \sum_{n=1}^{N} \tilde{w}(\mathbf{x}_n) $$ is unbiased. Even more importantly, the approximate posterior is : 

$$ 
\pi(\mathbf{x}) \approx \sum_{n=1}^{N} w(\mathbf{x}_n)\delta_{\mathbf{x}_n}(\mathbf{x}) \qquad w(\mathbf{x}_n) = \frac{\tilde{w}(\mathbf{x}_n)}{\sum_{k=1}^{N} \tilde{w}(\mathbf{x}_k)}
$$

Using normalized weights. If the normalizing constant was known exactly, then we could build a *non-normalized* IS estimator which is actually unbiased (with an almost equivalent derivation, omitted):  

$$

\widehat{\mathcal{I}}_{NN} := \frac{1}{N} \cdot \frac{1}{Z} \sum_{n=1}^{N}  f(\mathbf{x}_n) \frac{p(\mathbf{x}_n, \mathcal{D})}{q(\mathbf{x}_n)} = \frac{1}{N} \sum_{n=1}^{N}  f(\mathbf{x}_n) \frac{\pi(\mathbf{x}_n)}{q(\mathbf{x}_n)}

$$

Where $$Z$$ is the normalizing constant of the posterior distribution $$\pi(\mathbf{x})$$. In this post we are only concerned with self-normalized estimators which, while biased, turn out to have lower variance in several settings. 
From now on, an importance weight as function of sample $$ \mathbf{x}_n $$ is abbreviated as $$w_n$$; it is important however to keep in mind that weights are a function of inputs/samples. 
 
### Choice of proposal and variance of importance weights <a name="isproposal"></a>

It is pretty intuitive that our IS estimates can only be as good as our proposal. In general, we should seek a proposal that minimizes the variance of our estimators. This follows from the fact that the variance of a MC estimate (which is a sample average) is the expected square error from the true value of the integral. Let us see this by considering, for simplicity, the variance of the non-normalized estimator: 

$$ 
\mathbb{V}_{q} [ \widehat{\mathcal{I}}_{NN} ] =  \mathbb{E}_{q} \left [ \left ( \widehat{\mathcal{I}}_{NN} - \mathbb{E}_{q} \left [  \frac{f(\mathbf{x}) \pi(\mathbf{x})}{q(\mathbf{x})} \right ] \right )^{2} \right ] = \mathbb{E}_{q} \left [ \left ( \widehat{\mathcal{I}}_{NN} - \mathcal{I} \right )^2  \right ] 
$$

Which follows by simply applying the definition of variance (recalling that our samples are obtained through the proposal $$q$$ so that all expectations are under $$q$$). In order to derive the proposal that minimizes the variance, it is easier to inspect a different expression for the variance $$\mathbb{V}_{q} [ \widehat{\mathcal{I}}_{NN} ] $$, which uses the identity that variance equals second moment minus first moment squared, instead of the definition: 

$$ 

\mathbb{V}_{q} [ \widehat{\mathcal{I}}_{NN} ] = \frac{1}{N} \mathbb{V}_{q} \left [ \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right ] = \frac{1}{N} \mathbb{E}_q \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right )^2 \right ] - \frac{1}{N}  \underbrace{\left (  \mathbb{E}_q \left [ \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right ] \right )^2}_{=(\mathcal{I})^2}
$$

Notice that the term on the right in this expression is just $$\mathcal{I}^2$$ and thus does not involve $$q$$. We only need to minimize the first term with respect to $$q$$. Expanding this term on the left builds some intuition on what the form of the minimizing proposal looks like: 

$$\begin{equation}\begin{aligned}
\mathbb{E}_q \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right )^2 \right ]  &=  \int \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q(\mathbf{x})} \right )^2 q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \int \frac{(f(\mathbf{x})\pi(\mathbf{x}))^2}{q(\mathbf{x})} \mathrm{d}\mathbf{x} \\
&= \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right | \frac{\left | f(\mathbf{x})\pi(\mathbf{x})  \right |}{q(\mathbf{x})} \mathrm{d} \mathbf{x}
\end{aligned}\end{equation}\tag{15}\label{eq15}$$

Clearly, if we want this quantity to be small, then whenever the numerator is high $$q$$ should be at least as high. That is, when $$  \vert f(\mathbf{x}) \pi(\mathbf{x})  \vert $$ is high, then $$q(\mathbf{x})$$ should be high, or at least it should definitely not be small. That is, we need  $$  \vert f(\mathbf{x}) \pi(\mathbf{x})  \vert $$  large   $$\Rightarrow q(\mathbf{x})$$ large. In fact, the proposal that minimizes the variance turns out to be a normalized version of $$  \vert f(\mathbf{x}) \pi(\mathbf{x})  \vert $$  : 

$$ 
q^{*}(\mathbf{x}) = \frac{\left | f(\mathbf{x})\pi(\mathbf{x})  \right | }{\int \left | f(\mathbf{x})\pi(\mathbf{x})  \right |  \mathrm{d}\mathbf{x}}
$$

Unfortunately deriving this from scratch, by explicitly minimizing the second moment turns out to require some (basic) functional analysis. However, given knowledge of the optimal solution, we can verify that it is indeed minimizing the variance by making use of the following inequality:

$$
\mathbb{E}[x^2] \geq \mathbb{E}[ \left | x \right |]^{2}
$$

wich follows from the Cauchy-Schwartz inequality. We show that this bound is tight when using the optimal proposal. Plugging in the optimal proposal gives:

$$\begin{equation}\begin{aligned}
 \mathbb{E}_{q^{*}} \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q^{*}(\mathbf{x})} \right )^2 \right ] &=  \int   \left | f(\mathbf{x})\pi(\mathbf{x})  \right | \frac{\left | f(\mathbf{x})\pi(\mathbf{x})  \right |}{q^{*}(\mathbf{x})} \mathrm{d} \mathbf{x} \\
 &= \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right |  \mathrm{d}\mathbf{x} ~ \cdot ~ \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right |^2 \frac{1}{ \left | f(\mathbf{x})\pi(\mathbf{x})  \right | } \mathrm{d} \mathbf{x} \\
 &=  \left ( \int  \left | f(\mathbf{x})\pi(\mathbf{x})  \right |  \mathrm{d} \mathbf{x} \right )^2 
\end{aligned}\end{equation}\tag{16}\label{eq16}$$

which gives an expression for $$ \mathbb{E}_{q^{*}} \left [ \left ( \frac{f(\mathbf{x})\pi(\mathbf{x})}{q^{*}(\mathbf{x})} \right )^2 \right ] $$. Further:

$$\begin{equation}\begin{aligned}
 \mathbb{E}_{q^{*}} \left [ \left | \frac{f(\mathbf{x})\pi(\mathbf{x})}{q^{*}(\mathbf{x})} \right |^2 \right ] &= \left ( \int \left | \frac{f(\mathbf{x})\pi(\mathbf{x})}{q^{*}(\mathbf{x})} \right | q^{*}(\mathbf{x}) \mathrm{d} \mathbf{x}  \right )^2 \\
 &= \left ( \int \left | f(\mathbf{x})\pi(\mathbf{x}) \right |  \mathrm{d} \mathbf{x}  \right )^2
\end{aligned}\end{equation}\tag{17}\label{eq17}$$

which indeed gives the same expression and thus shows that the bound is tight.

### Sequential Importance Sampling <a name="sis"></a>

Let us now go back to the task of sequentially estimating a distribution of the form $$ \left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \}_{t}$$. This time however, we estimate any distribution by a set of weighted samples, a.k.a particles. 
Firstly, I am going to explain necessary notation. Note that the treatment in this section is very general and not specific to any particular state space model (hence not to the first order Markov one described earlier).  

* Let $$\gamma_{t}(\mathbf{s}_{1:t})$$ be the "target" distribution at time $$t$$ for states $$\mathbf{s}_{1:t}$$. Always keep track of all indices. For example, $$\gamma_{t}(\mathbf{s}_{1:t-1})$$ is a different object, namely $$\int \gamma_{t}(\mathbf{s}_{1:t}) \mathrm{d} \mathbf{s}_t $$. It is also different of course from $$\gamma_{t-1}(\mathbf{s}_{1:t-1})$$, which is simply the target at $$t-1$$. Importantly, note that the usual "target" is **the unnormalized version** of whatever our distribution of interest is ($$ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ or $$p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t}) $$ The reason we can ignore normalizing constants is that since these algorithms are IS based, we can always normalize the weights.
* Then, let $$\pi(\mathbf{s}_{1:t})$$ be the normalized version of the target $$\gamma_{t}(\mathbf{s}_{1:t})$$, i.e. : $$ \pi(\mathbf{s}_{1:t}) = \gamma_{t}(\mathbf{s}_{1:t}) / Z_t$$ with $$Z_t = \int \gamma_{t}(\mathbf{s}_{1:t}) \mathrm{d} \mathbf{s}_{1:t} $$
* The Dirac delta mass for multiple elements is defined naturally as $$\delta_{\mathbf{x}_{1:t}^{n}}(\mathbf{x}_{1:t}) :=  \delta_{\mathbf{x}_{1}^{n}}(\mathbf{x}_1) \delta_{\mathbf{x}_{2}^{n}}(\mathbf{x}_2) \dots \delta_{\mathbf{x}_{t}^{n}}(\mathbf{x}_t) $$
* While everything should be defined at some point, it is useful to keep in mind general principles such as whenever a symbol has a "hat" , that denotes an approximation, a "tilde" denotes an unnormalized quantity, and a $$\pi$$ a posterior.
* Useful to keep in mind: sometimes I will juggle between an importance weight $$w_{t}^{n}$$ that is specific to particle $$n$$ and what shoul really be called an importance weight *function* , that is the importance weight as a function of the state $$w_t = w_t(\mathbf{s}_t)$$. I will probably just call both "importance weight".

So, let's suppose then that we are trying to find a particle approximation for our target at iteration $$t$$: $$\gamma_{t}(\mathbf{s}_{1:t}) := p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$. We can use IS directly with a proposal distribution that also depends on $$\mathbf{s}_{1:t}$$ and find the  (unnormalized) importance weights: 

$$\begin{equation}\begin{aligned}
\tilde{w}_{t} = \frac{\gamma_t(\mathbf{s}_{1:t})}{\color{#FF8000}{q}_{t}(\mathbf{s}_{1:t})}
\end{aligned}\end{equation}\tag{18}\label{eq18}$$

With these , we can build the self-normalized importance sampling estimator as we have seen in the previous section. As we have seen in the discussion of IS, we can approximate the normalized posterior using normalized weights:

$$\begin{equation}\begin{aligned}
 p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \approx \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n}) \qquad \mathbf{s}_{1:t}^{n} \sim \color{#FF8000}{q}_{t}(\mathbf{s}_{1:t})
\end{aligned}\end{equation}\tag{19}\label{eq19}$$

where $$w_{t}^{n}$$ are the normalized weights, and we are using $$N$$ sample *trajectories* for our proposal. If we were only interested in $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) $$, we can simply discard previous samples: this is because  $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) $$ is just a marginal of $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) $$. Therefore, we can approximate the filtering distribution:

$$ 
p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) \approx \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{t}^{n}}(\mathbf{s}_t)
$$

So, how is this different to non-sequential importance sampling? The problem is that without explicitly stating any assumptions/constraints on the proposal these calculations scale linearly with the dimension of the state space $$t$$. It is intuitively unnecessary to propose a whole trajectory of samples at each iteration. Let's see how it is possible to avoid this by simply imposing a simple autoregressive (time series jargon) structure on the proposal. 
Let our new proposal at time $$t$$ be the product of two factors: 

$$
q_{t}\left(\mathbf{s}_{1:t}\right)=q_{t-1}\left(\mathbf{s}_{1:t-1}\right) q_{t}\left(\mathbf{s}_{t} | \mathbf{s}_{1:t-1}\right)
$$

In other words, to obtain a sample from the full proposal at time $$t$$, we can just take the previous trajectory that was sampled up to $$t-1$$ and append a sample from $$ q_t\left(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}\right)$$. We can now exploit this proposal structure to express the weights at time $$t$$ as a product between the previous weights at $$t-1$$ with a factor. The algebraic trick uses multiplying and dividing by the target at $$t-1$$ to artificially obtain the weights at $$t-1$$:

$$\begin{equation}\begin{aligned}
 \tilde{w}_{t}\left(\mathbf{s}_{1:t}\right) &=\frac{\gamma_{t}\left(\mathbf{s}_{1:t}\right)}{\color{#FF8000}{q}_{t}\left(\mathbf{s}_{1:t}\right)} \\ &=\frac{1}{\color{#FF8000}{q}_{t-1}\left(\mathbf{s}_{1:t-1}\right)} \frac{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right)}{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right)} \frac{\gamma_{t}\left(\mathbf{s}_{1:t}\right)}{\color{#FF8000}{q}_{t}\left(\mathbf{s}_{t} | \mathbf{s}_{1:t-1}\right)} \\ &=\frac{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right)}{\color{#FF8000}{q}_{t-1}\left(\mathbf{s}_{1:t-1}\right)} \frac{\gamma_{t}\left(\mathbf{s}_{1:t}\right)}{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right) \color{#FF8000}{q}_{t}\left(\mathbf{s}_{t} | \mathbf{s}_{1:t-1}\right)} \\
 &= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\gamma_{t}(\mathbf{s}_{1:t})}{\gamma_{t-1}(\mathbf{s}_{1:t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1})} := \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \varpi_{t}(\mathbf{s}_{t-1}, \mathbf{s}_t)
\end{aligned}\end{equation}\tag{20}\label{eq20}$$

Where we define the *incremental importance weight* $$\varpi_{t}(\mathbf{s}_{t-1}, \mathbf{s}_t)$$. It is a function of only current and previous states because, as we will see soon, $$\gamma_t$$ decomposes recursively and the $$\gamma_{t-1}$$ terms cancel, leaving only terms that depend on $$\mathbf{s}_t,\mathbf{s}_{t-1}$$. Therefore, we can approximate our desired distribution as:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \approx \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n})
\end{aligned}\end{equation}\tag{21}\label{eq21}$$

with the weights $$w_{t}^{n}$$ defined as the normalized weights found in \eqref{eq15}. 
It is very important to notice that in the key equation defining SMC algorithms \eqref{eq20} one is performing IS in the *joint* space $$\mathbf{s}_{1:t}$$. In other words, we are performing inference using the TFD targeting $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$, not the SFD, because we can just estimate integrals wrt $$p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t})$$ by discarding samples. This will turn out to be relevant in later sections.

As shown in the IS section, we can approximate the normalizing constant as:

$$
\widehat{Z}_t = \frac{1}{N} \sum_{n=1}^{N} \tilde{w}_{t}^{n} = \frac{1}{N} \sum_{n=1}^{N} \prod_{k=1}^{t} \varpi_k(\mathbf{s}_{k-1}^{n}, \mathbf{s}_{k}^{n})
$$

This is the essence of SIS (Sequential Importance Sampling). Important note: this is a standard presentation you can find e.g. from Doucet et al [2]. However, you should note that for example, if we put this into context of state space models say, then the proposal can depend on measurements too. Crucially, although it would be natural to split the proposal as: $$ \color{#FF8000}{q}_{t}\left(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}\right)= \color{#FF8000}{q}_{t-1}\left(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:\color{red}{t-1}}\right) \color{#FF8000}{q}_{t}\left(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:\color{red}{t}}\right)$$ this is usually a *choice*, and we could make both terms dependent on the current measurements! We will come back to this when discussing the Auxiliary Particle Filter. 

Ok, now it's time to apply SIS to the state space model we covered earlier. In this context, what we want is again $$\left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \}_{t} $$ , hence our target $$\gamma$$ is the unnormalized posterior: $$\gamma_{t}(\mathbf{s}_{1:t}) := p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$. Keep in mind that we can always get the filtering distribution from $$\left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \}_{t} $$. Now the recursion that we developed earlier in the post for the joint $$ p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$ becomes useful in deriving the weight update for SIS: 

$$\begin{equation}\begin{aligned}
\varpi_{t}(\mathbf{s}_{t-1}, \mathbf{s}_{t}) &= \frac{\gamma_{t}(\mathbf{s}_{1:t})}{\gamma_{t-1}(\mathbf{s}_{1:t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})} \\
&=  \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) \overbrace{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})}^{\cancel{\gamma_{t-1}(\mathbf{s}_{1:t-1})}}}{\cancel{\gamma_{t-1}(\mathbf{s}_{1:t-1})} \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})} \\
&=  \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})}
\end{aligned}\end{equation}\tag{22}\label{eq22}$$$$


Where in the conditioning of the proposal we introduce dependence on all measurements (usually we only use the latest). If you are given a choice for the proposal $$\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t}) $$, then you have a concrete algorithm to sequentially approximate $$\left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \}_{t \geq 1}$$, with constant time per update (remembering that throughout the algorithm only uses unnormalized weights, and only when one wants to approximate the desired distribution one needs to normalize the weights). This algorithm is neat, but it can be shown that the variance of the resulting *estimates* increases expontentially in $$t$$.
An important tangent is necessary at this point. In IS, we analysed the variance of estimators for integrals under the distribution of interest. In SIS, it makes more sense to focus on the variance of the importance weights, rather than the variance of some moments (integrals) under the TFD or SFD. This is because we don't know exactly which integrals we would be interested in, and it is easy to derive cases where the variance of some specific moment is low, but higher on any other. 

Closed this brief tangent, the exponentially increasing is due to SIS being a special case of IS.
To check this , consider the variance of $$\widehat{Z}/ Z_t $$ known as "relative variance" under simple IS: 

$$\begin{equation}\begin{aligned}
\mathbb{V}_q\left[ \frac{\widehat{Z}_t}{Z_t} \right] &=  \frac{\mathbb{V}_q[\widehat{Z}_t]}{Z_{t}^{2}} \qquad \text{since}~Z_t~ \text{a constant} \\ 
&= \frac{\frac{1}{N^2}\sum_{n=1}^{N} \mathbb{V}_q[\tilde{w}_{t}^{n}]  }{Z_{t}^{2}} \qquad \text{since weights are uncorrelated}\\
&= \frac{\frac{1}{N^2}\sum_{n=1}^{N} \mathbb{V}_q \left [\frac{\gamma_t(\mathbf{s}_{1:t})}{q_t(\mathbf{s}_{1:t})} \right ]  }{Z_{t}^{2}} \\
&=  \frac{\frac{1}{N^2}\sum_{n=1}^{N} \left \{ \mathbb{E}_q \left [ \left ( \frac{\gamma_t(\mathbf{s}_{1:t})}{q_t(\mathbf{s}_{1:t})} \right )^2 \right ] - \left (\mathbb{E}_q \left [ \frac{\gamma_t(\mathbf{s}_{1:t})}{q_t(\mathbf{s}_{1:t})} \right ] \right )^2 \right \} }{Z_{t}^{2}} \\ 
&= \frac{\frac{1}{N^2}\sum_{n=1}^{N} \left \{ \int \frac{(\gamma_t(\mathbf{s}_{1:t}))^2}{(q_t(\mathbf{s}_{1:t}))^2}  q_t(\mathbf{s}_{1:t})\mathrm{d}\mathbf{s}_{1:t} - \left (\int  \frac{\gamma_t(\mathbf{s}_{1:t})}{q_t(\mathbf{s}_{1:t})} q_t(\mathbf{s}_{1:t})\mathrm{d}\mathbf{s}_{1:t} \right )^2 \right \}}{Z_{t}^{2}} \\
&= \frac{\frac{1}{N^2}\sum_{n=1}^{N} \left \{ \int \frac{(\gamma_t(\mathbf{s}_{1:t}))^2}{q_t(\mathbf{s}_{1:t})} \mathrm{d}\mathbf{s}_{1:t} - \left (\int  \gamma_t(\mathbf{s}_{1:t})\mathrm{d}\mathbf{s}_{1:t} \right )^2 \right \}}{Z_{t}^{2}} \\
&=  \frac{\frac{1}{N^2} \cdot N \cdot  \int \frac{(\gamma_t(\mathbf{s}_{1:t}))^2}{q_t(\mathbf{s}_{1:t})} \mathrm{d}\mathbf{s}_{1:t} }{Z_{t}^{2}} - \frac{ \frac{1}{N^2}\cdot N \cdot  \overbrace{\left (\int  \gamma_t(\mathbf{s}_{1:t})\mathrm{d}\mathbf{s}_{1:t} \right )^2}^{Z_{t}^2}}{Z_{t}^{2}} \\ 
&= \frac{1}{N}\left (  \int  \frac{(\gamma_t(\mathbf{s}_{1:t}))^2}{Z_{t}^{2} q_t(\mathbf{s}_{1:t})}  \mathrm{d}\mathbf{s}_{1:t} - 1 \right ) = \frac{1}{N}\left (  \int  \frac{(\pi_t(\mathbf{s}_{1:t}))^2}{ q_t(\mathbf{s}_{1:t})}  \mathrm{d}\mathbf{s}_{1:t} - 1 \right )
\end{aligned}\end{equation}\tag{23}\label{eq23}$$$$

We now show that even for an extremely simple model, this expression is exponential in $$t$$. This example is taken from Doucet et al. [2]. Consider a univariate state space model where the TFD at each timestep is a Gaussian. Then, the sequence of normalized and unnormalized target distributions, and normalizing constant at time $$t$$ are:  

$$
\gamma_t(s_{1:t}) = \prod_{k=1}^{t} \exp \left ( -\frac{1}{2} s_{k}^{2}  \right ) \qquad Z_t = (2\pi)^{t/2}
$$

Or in other words $$\pi_t(s_{1:t}) = \prod_{k=1}^{t} \mathcal{N}(s_k \mid 0, 1) = \mathcal{N}(s_{1:t} \mid \boldsymbol{0}, \mathbf{I})$$. Suppose we select a simple proposal distribution as a factorised Gaussian with unknown variance: 

$$
q_t(s_{1:t}) = \prod_{k=1}^{t} q_{k}(s_k) = \prod_{k}^{t} \mathcal{N}(s_k \mid 0, \sigma^2) = \mathcal{N}(s_{1:t} \mid \boldsymbol{0}, \sigma^2 \mathbf{I})
$$

Then, :
$$\begin{equation}\begin{aligned}
\mathbb{V}_q\left[ \frac{\widehat{Z}_t}{Z_t} \right] &= \frac{1}{N} \left [ \int   \frac{\left ( \prod_{k=1}^{t} \mathcal{N}(s_k \mid 0,1) \right)^2}{\prod_{k=1}^{t} \mathcal{N}(s_k \mid 0,\sigma^2)} \mathrm{d}s_{1:t} - 1\right] \qquad \text{directly from 23} \\
&= \frac{1}{N} \left [ \int   \frac{(2\pi)^{-t} \left (\prod_{k=1}^{t}  \exp \left\{ -\frac{1}{2}s_{k}^{2} \right\}\right ) \left (\prod_{k=1}^{t}  \exp \left\{ -\frac{1}{2}s_{k}^{2} \right\}\right )}{\prod_{k=1}^{t} (2\pi \sigma^2)^{-1/2} \exp \left\{ -\frac{1}{2\sigma^2} s_{k}^2 \right\}} \mathrm{d}s_{1:t} - 1\right] \\
&= \frac{1}{N} \left [\frac{(2\pi)^{-t}}{(2\pi \sigma^2)^{-t/2}} \int   \frac{ \exp\left\{ -\sum_{k=1}^{t}s_{k}^2 \right\} }{\exp \left\{ -\frac{1}{2\sigma^2}\sum_{k=1}^{t}s_{k}^{2} \right\}} \mathrm{d}s_{1:t} - 1\right] \\
&= \frac{1}{N} \left [\frac{(2\pi \sigma^2)^{t/2}}{(2\pi)^t} \int  \exp \left\{ -\sum_{k=1}^{t}s_{k}^2 + \frac{1}{2\sigma^2} \sum_{k=1}^{t}s_{k}^2 \right\} \mathrm{d}s_{1:t} - 1\right] \\
&= \frac{1}{N} \left [\frac{(2\pi \sigma^2)^{t/2}}{(2\pi)^t} \int  \exp \left\{ \left ( -\frac{1}{2}\left [ 2 - \frac{1}{\sigma^2} \right ] \right ) s_{1:t}^{\top} s_{1:t}  \right\} \mathrm{d}s_{1:t} - 1\right] \qquad \text{as}~ s_{1:t}^{\top}s_{1:t} = \sum_{k=1}^{t} s_{k}^{2} \\
&= \frac{1}{N} \left [\frac{(2\pi \sigma^2)^{t/2}}{(2\pi)^t} \cdot \left ( 2\pi \cdot \frac{\sigma^2}{2\sigma^2 -1 } \right)^{t/2} - 1\right] \qquad \text{using}~ \left [ 2 - \frac{1}{\sigma^2} \right ]^{-1} = \left [\frac{\sigma^2}{2\sigma^2 -1} \right ] \\
&= \frac{1}{N} \left [\frac{\cancel{(2\pi)^{t/2}} \sigma^t }{\cancel{(2\pi)^t}} \cdot  \cancel{(2\pi)^{t/2}} \left ( \cdot \frac{\sigma^2}{2\sigma^2 -1 } \right)^{t/2} - 1\right] \\
&= \frac{1}{N} \left [(\sigma^2)^{t/2} \cdot   \left ( \frac{\sigma^2}{2\sigma^2 -1 } \right)^{t/2} - 1\right] \\
&= \frac{1}{N} \left [\left ( \frac{\sigma^4}{2\sigma^2 -1 } \right)^{t/2} - 1\right]
\end{aligned}\end{equation}\tag{24}\label{eq24}$$

For example, if $$\sigma^2 = 1.2$$, then $$N \cdot \mathbb{V}_q\left[ \frac{\widehat{Z}_t}{Z_t} \right] \approx (1.103)^{t/2}$$, which for sequence length $$t=1000$$ equals $$1.9 \cdot 10^{21} $$. In this case, to have a small relative variance, say $$ 0.01$$, we would need $$N \approx 2 \cdot 10^{23}$$ particles which is obviously infeasible.

The exponentially increasing variance has other negative consequences, the first of which is known under the names of *sample degeneracy* or *weight degeneracy*. Basically, if you actually run this after not-so-many iterations there will be one weight $$\approx 1$$ and all other will be zero, which equates to approximate the target with one sample. 

![hhm]({{ '/assets/images/sample-deg.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2: Sample or Weight degeneracy of SIS. The size of disks represent the size of the corresponding weight to a particle. Borrowed from Naesseth et al. [4]*


### Resampling <a name="resampling"></a> 

This is where Sequential Monte Carlo (SMC) or Sequential Importance Resampling (SIR)/ particle filtering algorithms come into the picture. They mitigate the weight degeneracy issue by explicitly changing the particle set that was found in the previous iteration. They do so by resampling independently with replacement an equally sized particle set, where each sample is sampled with probability equal to its weight. This particular type of resampling is equivalent to sampling from a multinomial distribution with parameters equal to the weights, and is thus called multinomial resampling. Thus, we could see SMC/SIR as the same algorithm as SIS with an added step at the end of each iteration, where we resample particles according to their weights, and modify these afterwards to be $$1/N$$. At a high level, resampling is often motivated as a tool to eliminate particles with low weights and multiply those with high weights, so as to focus our computational resources in the most important parts of the space. In essence, adding a resampling step at the end of each iteration of SIS becomes SMC.

The resampling step can be intepreted as a clever choice of proposal. To understand this, one needs to know that sampling from a mixture can be achieved via multinomial resampling with weights equal to the mixture weights. Consider the first iteration of SIS. We have sampled particles from a proposal $$\left \{ \mathbf{s}_{1}^{n} \right \}_{n=1}^{N} \sim \color{#FF8000}{q}_{1}(\mathbf{s}_{1})$$ and calculated corresponding weights. An approximation to the (normalized) target, as we have already shown, is $$ \pi_1(\mathbf{s}_1) \approx \widehat{\pi}_{1} = \sum_{n=1}^{N} w_{1}^{n} \delta_{\mathbf{s}_{1}^{n}} (\mathbf{s}_1) $$. Now, instead of "propagating" particles to the next iteration by sampling them from $$ \color{#FF8000}{q}_{2}(\mathbf{s}_{2} \mid \mathbf{s}_1) $$, we use the information gathered in the previous iteration, compressed in $$\widehat{\pi}_{1}$$, and sample the trajectory $$\mathbf{s}_{1}, \mathbf{s}_2 $$ from $$ \widehat{\pi}_{t} \cdot \color{#FF8000}{q}_{1}(\mathbf{s}_{2} \mid \mathbf{s}_1) $$ instead. This is the same as resampling the particles at the end of iteration $$t=1$$, and sampling the new particles at $$t=2$$ from the proposal *evaluated at the resampled particles*.

<div id="example1">
  <i> <b> (Meta) Algorithm 1: Sequential Monte Carlo / Sequential Importance Resampling </b> </i> <br>


  At time $t=1$: 
  <ol>
    <li> <b>Propagation</b> : sample from proposal $\mathbf{s}_{1}^{n} \sim {\color{#FF8000}q}_{1}(\mathbf{s}_1)$ </li>
    <li> <b>Update</b>: compute weights as $w_{1}^{n} \propto \varpi_{1}^{n}(\mathbf{s}_{0}^{n}, \mathbf{s}_{1}^{n})$  </li>
    <li> <b>Resample</b>: $\left \{ \mathbf{s}_{1}^{n} , w_{1}^{n} \right \}_{n=1}^{N} $ to obtain $ \left \{ \mathbf{r}_{1}^{n}, 1/N \right \}_{n=1}^{N} $ </li>
  </ol>

  At time $t \geq 2$:
  <ol>
    <li> <b>Propagation</b> : sample from proposal $\mathbf{s}_{t}^{n} \sim {\color{#FF8000}q}_{t}(\mathbf{s}_{t} \mid \mathbf{r}_{1:t-1}^{n})$ and set $ \mathbf{s}_{1:t}^{n} \leftarrow (\mathbf{r}_{1:t-1}^{n}, \mathbf{s}_{t}^{n})$ </li>
    <li> <b>Update</b>: compute weights as $w_{t}^{n} \propto \varpi_{t}^{n}(\mathbf{s}_{t-1}^{n}, \mathbf{s}_{t}^{n})$ </li>
    <li> <b>Resample</b>: $\left \{ \mathbf{s}_{1:t}^{n} , w_{t}^{n} \right \}_{n=1}^{N} $ to obtain $ \left \{ \mathbf{r}_{1:t}^{n}, 1/N \right \}_{n=1}^{N} $  </li>
 </ol>
</div>

<br>

Where I use $$\mathbf{r}$$ to emphasize that a particle has been resampled. Notice that the weight computation does not involve the previous weight, since resampling sets weights to a constant, and thus we can omit it when using proportionality.

In our state space model, if we chose a proposal equal to the transition density, so $$ \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t}) = \color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})$$ , then the weight update simplifies to: 

$$\begin{equation}\begin{aligned}
\varpi_t(\mathbf{s}_{t-1}, \mathbf{s}_{t}) &=  \frac{\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)}{\color{#FF8000}{q}_t(\mathbf{s}_t \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})} \\
&= \frac{\cancel{\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1})} \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)}{\cancel{\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1})}} \\
&= \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)
\end{aligned}\end{equation}\tag{25}\label{eq25}$$

This choice within the SMC general framework gives rise to the concrete algorithm named *Bootstrap Particle Filter* (BPF). Using the transition density can lead to poor approximations, especially if the dimension of the hidden states is large. 

Let us now discuss some details about $$ \widehat{\pi}_{t}$$ . This paragraph can probably be skipped at first reading. In vanilla SIS, at each iteration we build an approximate empirical distribution to our target, say $$ \pi_t(\mathbf{s}_{1:t}) = p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$, that is represented by a weighted sum or "mixture" : $$  \widehat{\pi}_{t} = \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n})$$. The samples $$\left \{ \mathbf{s}_{1:t}^{n} \right \}_{n=1}^{N}$$ used in our approximation however come from the proposal $$ q_t(\mathbf{s}_{1:t})$$. We can obtain a set of samples approximately distributed according to $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ by sampling from our mixture approximation. Keep in mind the fact that sampling from a mixture is equivalent to resampling. Crucially, using these resampled particles, we can form a *different* estimator for $$\pi_t(\mathbf{s}_{1:t}) $$ than $$ \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n})$$, namely: $$ \frac{1}{N} \sum_{n=1}^{N}\delta_{\mathbf{r}_{1:t}} (\mathbf{r}_{1:t}^{n})$$ where $$\left \{ \mathbf{r}_{1:t} \right \}_{n=1}^{N} $$ is the "resampled" particle trajectory (resampled in the sense that, to sample from the mixture approximation, one uses resampling). It is worth to restate that this is because these new samples actually come (approximately) from $$ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ and thus to approximate the distribution itself we use its empirical approximation. It can be shown that the estimates (e.g. moments) under this approximation generally have *higher variance* than under the previous estimator. This is the price that we have to pay to mitigate the weight degeneracy: increase (at least, temporarily) the variance of the estimator to however reduce it in the long run. 


![pf]({{ '/assets/images/bpff.svg' | relative_url }})
{: style="width: 100%;" class="center; height: 100%"}
*Fig. 3: Illustration of BPF for a one dimensional set of particles. Tikz figure with minor modifications from an original made by Víctor Elvira.*


Unfortunately, weight degeneracy is not the end of the story. The resampling mechanism generates another important issue known as *path degeneracy*. This phenomenon is best understood with an illustration, shown below. Repeated resampling over many iterations causes particle diversity to be killed, as most of the particles at some point will collapse back to a single (or few) ancestor. 

![hhm]({{ '/assets/images/path-degg.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4: Path degeneracy illustration in SMC/SIR/PF. Borrowed from Naesseth et al. [4]*

There are ways to deal with path degeneracy, such as low-variance resampling, or simply only resampling when a certain measure of efficiency is satisfied (effective sample size), or adaptive resampling. We do not go into more detail here, as these are more advanced, and path degeneracy is still not fundamentally solved. For a survey of resampling strategies, see [11].

## Propagating particles by incorporating the current measurement <a name="apf"></a>

Let's talk about one particularly popular variation on the BPF. I will put it into the context of a generic SMC algorithm (for state space models), as did for the BPF, and explain different intepretations. I will start with the interpretation given by Doucet et al [2].

One of the motivations for APF is as follows. One can show that, if we had acces to the locally optimal proposal in SMC, then the weight update becomes an expression that does not involve the current state $$\mathbf{s}_t$$ at all. In fact, we will show this in the next subsection. Notice that previously we have been performing propagation first, then weight update and resampling. Now, if the weight update does not depend on the current state, nothing would stop us at performing resampling before propagation. As Doucet et al [2] point out, this yields a better approximation of the distribution as it provides a greater number of distinct particles to approximate the target. 

This interchange of propagation and resampling can be seen as a way of incorporating the effects of the current measurements $$\mathbf{v}_t$$ on the generation of states at time $$t$$, or of "filtering out" particles with low importance. 
Because in general we don't have access to the optimal proposal and thus can't do this exactly, we can try to "mimic" it, and this is what APF attempts to do. 

Before getting into APF however, let's actually inspect what would happen if we used the locally optimal proposal in filtering. 

### The effect of using the locally optimal proposal <a name="optimalproposal"></a>

In the context of the state space model described, the following proposal is often referred to as the "optimal" or "locally optimal" proposal: 

$$ 
\color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t}) = p(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}, \mathbf{v}_t) = \frac{p(\mathbf{v}_{t} \mid \mathbf{s}_{t}, \mathbf{s}_{t-1}) p(\mathbf{s}_{t} \mid \mathbf{s}_{t-1})}{p(\mathbf{v}_t \mid \mathbf{s}_{t-1})} = \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) \color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})}{\int \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) \color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \mathrm{d}\mathbf{s}_t}
$$

Applying Bayes' rule on $$ \mathbf{s}_t , \mathbf{v}_t$$ with $$\mathbf{s}_{t-1}$$ as "context". This name is due to the fact that it is the proposal that minimizes the variance of the weights (we have seen that this makes more sense than trying to minimize the variance of some moments under the posterior). Then, the weight update becomes: 

$$\begin{equation}\begin{aligned}
\varpi_{t}(\mathbf{s}_{t-1}, \mathbf{s}_{t}) &= \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_t)  }{ p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})} } \\
&=  p(\mathbf{v}_t \mid \mathbf{s}_{t-1})
\end{aligned}\end{equation}\tag{26}\label{eq26}$$

Since this expression does not depend on the current state $$\mathbf{s}_t$$, which has been integrated out, intuitively the (conditional) variance of the weights under the proposal at time $$t$$ is just $$0$$. 
To see this more explicitly:

$$
\mathbb{V}_{\color{#FF8000}{q}} \left[ \overbrace{\frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1})}{\color{#FF8000}{q}_t(\mathbf{s}_t \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})}}^{\varpi_t} \right] = \mathbb{E}_{\color{#FF8000}{q}} \left[ \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)^2\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1})^2}{\color{#FF8000}{q}_{t}(\mathbf{s}_t \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})^2} \right] - \left ( \mathbb{E}_{\color{#FF8000}{q}} \left[  \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1})}{\color{#FF8000}{q}_t(\mathbf{s}_t \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})} \right] \right )^2 = \int \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t)^2\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1})^2}{\color{#FF8000}{q}_t(\mathbf{s}_t \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:t})} \mathrm{d}\mathbf{s}_t  - p(\mathbf{v}_t \mid \mathbf{s}_{t-1})^2
$$

Where indeed if plugging in the optimal proposal for $$\color{#FF8000}{q}_t$$ gives 0.

The two main difficulties that using this proposal presents are:
1. Sampling from it is just as hard as sampling from $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$
2. It requires evaluation of $$ p(\mathbf{v}_t \mid \mathbf{s}_{t-1}) = \int \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}) \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}) \mathrm{d} \mathbf{s}_t$$ which is almost always an integral as difficult as the filtering problem itself.

### The Auxiliary Particle Filter <a name="apf2"></a>

#### A first intepretation: a standard SMC algorithm with a different $$\gamma$$ <a name="firstapf"></a>
In general, the APF can be thought of as a class of methods, within SMC, that notices that it would make sense, before propagating the particles, to immediately utilize $$\mathbf{v}_t$$, and get rid of unlikely particles. This echoes attempting to use the optimal proposal, since it is of the form $$p(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{v}_t)$$. Because of this "look ahead", APF tends to perform much better than the BPF when the likelihood is particularly informative.

The APF can be interpretated as a standard SMC algorithm (that is, an instantiation of the "meta" algorithm we described previously) where the target $$\gamma$$ that is propagated through each iteration is *not* the unnormalized filtering distribution $$p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t}) $$, but rather $$\gamma_t(\mathbf{s}_{1:t}) = p(\mathbf{s}_{1:t}, \mathbf{v}_{1:\color{red}{t+1}}) $$. This is how it achieves the incorporation of the next measurements before propagation.  

Under this interpretation, the target in the APF can be easily decomposed as:

$$\begin{equation}\begin{aligned}
\gamma_{t}(\mathbf{s}_{1:t}) := p(\mathbf{s}_{1:t} , \mathbf{v}_{\color{red}{t+1}}) &= \int p(\mathbf{s}_{1:t+1} , \mathbf{v}_{1:t+1}) \mathrm{d} \mathbf{s}_{t+1} \\
&= \int p(\mathbf{s}_{1:t} , \mathbf{v}_{1:t}) \cdot {\color{blue}{f}}(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}) \cdot {\color{green}{g}}(\mathbf{v}_{t+1} \mid \mathbf{s}_{t+1}) \mathrm{d} \mathbf{s}_{t+1} \\
&=  p(\mathbf{s}_{1:t} , \mathbf{v}_{1:t}) \int {\color{blue}{f}}(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}) \cdot {\color{green}{g}}(\mathbf{v}_{t+1} \mid \mathbf{s}_{t+1}) \mathrm{d} \mathbf{s}_{t+1} \\
&=  p(\mathbf{s}_{1:t} , \mathbf{v}_{1:t}) \cdot \underbrace{p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})}_{"predictive~likelihood"} 
\end{aligned}\end{equation}\tag{27}\label{eq27}$$

Which we see is equivalent to the product between what would be the target in standard SMC times the so called "predictive likelihood" $$p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})$$. The weight update can be derived by making use of this: 

$$\begin{equation}\begin{aligned}
\varpi_t(\mathbf{s}_{t-1}, \mathbf{s}_t) &=  \frac{\gamma_{t}(\mathbf{s}_{1:t})}{\gamma_{t-1}(\mathbf{s}_{1:t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1})} \\
&=    \frac{p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t}) p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})}{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})} \\
&=  \frac{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})} \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})}{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})} p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})} 
\end{aligned}\end{equation}\tag{28}\label{eq28}$$


Suppose we have been running the APF for $$1\dots t-1$$ iterations, and now we want a particle approximation of $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$. We can't compute the weights as in APF, because recall that it sequentially estimates something different to what we want, namely $$p(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:t}) \cdot p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})$$. Thus, to compute the weights for our approximation, we have to use $$\gamma_t(\mathbf{s}_{1:t}) = p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$ and $$\gamma_{t-1}(\mathbf{s}_{1:t-1}) = p(\mathbf{s}_{1:t-1} , \mathbf{v}_{1:t-1}) \cdot p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})$$. Doing the whole derivation: 



$$\begin{equation}\begin{aligned}
\varpi_t(\mathbf{s}_{t-1}, \mathbf{s}_t) &=  \frac{p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})}{ p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})}  \\
&=  \frac{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})}\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})} p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})} \\
&=  \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})}
\end{aligned}\end{equation}\tag{29}\label{eq29}$$ 


Please keep in mind that here we are talking about the importance weight used to estimate $$\gamma_t$$, while for all other previous targets we used \eqref{eq28}. This intepretation sees the previous targets $$\gamma_1,\dots,\gamma_{t-1}$$ as some sort of "bridging" densities of aid in the sequential propagation of the particles. Note that in practice the predictive likelihood involves an intractable integral, so we have to approximate it with $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) $$. However, in the ideal case, selecting $$\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}) =  p(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})$$ and $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) = p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})$$ leads to the so called "perfect adaptation" . 

Setting the approximation to the predictive likelihood to $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) = \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}(\mathbf{s}_{t})) $$ where $$ \boldsymbol{\mu}(\mathbf{s}_{t})$$ is some likely value is common. For example , if we choose as approximation to the predictive likelihood : $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) = \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}) $$  where $$\boldsymbol{\mu}_{t}$$ is the mean of $$ f(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) $$ *and* we also choose $$ \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1}) = f(\mathbf{s}_{t} \mid \mathbf{s}_{t-1})$$ , then we recover as special case the popular version of the APF weights: 

$$\begin{equation}\begin{aligned}
\varpi_t(\mathbf{s}_{t-1}, \mathbf{s}_t) &=  \frac{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})} \\
&=  \frac{\cancel{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})} \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}) \cancel{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})}}
\end{aligned}\end{equation}\tag{30}\label{eq30}$$ 

#### The original intepretation of APF and Marginal Particle Filters <a name="marginalpf"></a>
Up until now, we have derived concrete instatiations of particle filtering algorithms by performing importance sampling in the joint space (or trajectory space) (see \eqref{eq18}, \eqref{eq20}). In other words, we could also see this as "using" the TFD in \eqref{eq9}, rather than the SFD \eqref{eq10}: we motivated this initially by the fact that we can get an estimate of $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t})$$ from an estimate of $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ by ignoring previous samples. However, if we were only ever interested in $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t})$$, this approach isn't the best : the target distribution grows in dimension at each step, and this is partly why we need to perform resampling to reduce variance. In Marginal PFs [5], we perform importance sampling in the marginal space, that is with target distribution $$ p(\mathbf{s}_t \mid \mathbf{v}_{1:t})$$; computing importance weights in this way, however, increases the computational cost of the algorithm from $$\mathcal{O}(N)$$ to $$\mathcal{O}(N^2)$$. This is because of the different importance weight computation. Since now we are using SFD, the target distribution is proportional to: 

$$
p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) \propto \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t ) p(\mathbf{s}_t \mid \mathbf{v}_{1:t-1}) = \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t ) \int \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d} \mathbf{s}_{t-1} \approx \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t ) \sum_{n=1}^{N} w_{t-1}^{n} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n})  
$$

where crucially we use the particle approximation of the filtering distribution at time $$t-1$$: $$p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \approx \sum_{n=1}^{N} w_{t-1}^{n} \delta_{\mathbf{s}_{t-1}}(\mathbf{s}_{t-1}^{n})$$. Now, notice that the target can be rewritten as: 

$$
\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t ) \sum_{n=1}^{N} w_{t-1}^{n} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n}) = \sum_{n=1}^{N} w_{t-1}^{n} p(\mathbf{v}_t \mid \mathbf{s}_{t-1}^{n}) p(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}^{n}, \mathbf{v}_t)
$$

recalling the expression for the optimal proposal $$ p(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{v}_t) $$. I think this simple rearragement is interesting: firstly, now all the terms depend on the previous states $$\mathbf{s}_{t-1}^{n}$$; secondly, it perfectly shows the conditions that we want to satisfy for a good proposal (more on this soon). Now that we have the target, in other words the numerator of the importance weight, we are free to choose any proposal distribution we want. Recall that we are not in the setting of the autoregressive proposal of \eqref{eq18}, \eqref{eq20}; the proposal now is simply a function of $$\mathbf{s}_t$$. It makes sense to choose a proposal that has the same structure as the numerator (as we are trying to match it), that is:

$$
\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{1:t}, \mathbf{s}_{t-1}) = \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1}) = \sum_{n=1}^{N} w_{t-1}^{n} \color{#FF8000}{q}_{t}(\mathbf{s}_t \mid \mathbf{v}_t, \mathbf{s}_{t-1}^{n})
$$

i.e. a mixture proposal, so that the unnormalized importance weight is computed as:

$$
\widetilde{w}_{t} = \frac{ \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t ) \sum_{n=1}^{N} w_{t-1}^{n} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n})}{ \sum_{n=1}^{N} w_{t-1}^{n} \color{#FF8000}{q}_{t}(\mathbf{s}_t \mid \mathbf{v}_t, \mathbf{s}_{t-1}^{n})}
$$

This equation resembles the form of \eqref{eq22} (standard importance weight for state space models): we can see this new importance weight as obtained from \eqref{eq22} by marginalizing previous states. This is quite neat I think. 
The question is, what should $$ \color{#FF8000}{q}_{t}(\mathbf{s}_t \mid \mathbf{v}_t, \mathbf{s}_{t-1}^{n})$$ be ? 
We have seen that the mixture in the denominator needs to approximate both the predictive likelihood $$ p(\mathbf{v}_t \mid \mathbf{s}_{t-1}^{n}) $$ and the optimal proposal $$p(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}^{n}, \mathbf{v}_t)$$ (both expressions that cannot be evaluated exactly) *for all* $$n$$. Recall that $$p(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}^{n}, \mathbf{v}_t) \propto  \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_t) \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n})$$. We can then attempt this in different ways: for example, we could just set $$\color{#FF8000}{q}_{t}(\mathbf{s}_t \mid \mathbf{v}_t, \mathbf{s}_{t-1}^{n})$$ to be the transition kernels $$\color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n})$$ . In this case , we would need to multiply $$w_{t-1}^{n}$$ by some other term in order to match the predictive likelihood term in the numerator, as well as the likelihood $$\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_t)$$ . In this case, all the work goes into the choice of what to multiply $$w_{t-1}^{n}$$ and $$ \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n})$$ within the sum. 

Of course, since $$\widetilde{w}_t$$ is a function of $$\mathbf{s}_t$$, we now have to compute these sums for all particles, which gives the cost of $$\mathcal{O}(N^2)$$. Something else that can immediately be seen is that this expression, in a sense, is just more general than \eqref{eq22}, which can be recovered simply replacing the sums with a single term. This is the basic idea behind the Multiple Importance Sampling interpretation of PFs, which is more or less the same as marginal particle filtering, but where certain things are presented more explicit.

I think it is quite natural to introduce how APF was originally motivated after having explained MPFs and especially the discussion on the two differrent expressions for the numerator of the unnormalized SFD $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t})$$. In the original paper [11], Pitt and Shepard look at the numerator of the SFD and consider as target distribution the following joint:

$$
p(n, \mathbf{s}_t \mid \mathbf{v}_{1:t} ) \propto \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t )  w_{t-1}^{n} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n}) = w_{t-1}^{n} p(\mathbf{v}_t \mid \mathbf{s}_{t-1}^{n}) p(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}^{n}, \mathbf{v}_t) 
$$

where $$n$$ is an *index of the full mixture*, and is an auxiliary variable which is where the name of the algorithm actually comes from. Little *concrete* interpretation is provided for this choice in the paper, but we will see in the next section how this is casted as an implicit assumption. From this joint, the marginal of the index is: 

$$
p(n \mid \mathbf{v}_{1:t}) \propto w_{t-1}^{n} p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}^{n})
$$

As we know the predictive likelihood defines an intractable integral: a common approximation we have seen is $$\color{green}{g}(\mathbf{v}_t \mid \boldsymbol{\mu}_{t}^{n})$$.  Then, define the probability of the index to be the "simulation weight" or "preweight" : $$p(n \mid \mathbf{v}_{1:t}):= \lambda_{t}^{n}  \propto w_{t-1}^{n} \color{green}{g}(\mathbf{v}_t \mid \boldsymbol{\mu}_{t}^{n}) $$. 

Using this, we construct a proposal with the same form of the target, which is now $$p(n, \mathbf{s}_t \mid \mathbf{v}_{1:t} )$$:

$$
q_t(n, \mathbf{s}_t \mid \mathbf{v}_{1:t}) = q_t(n \mid \mathbf{v}_{1:t}) q_t(\mathbf{s}_t \mid \mathbf{v}_{1:t}) = \lambda_{t}^{n} q_t(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n}, \mathbf{v}_t)
$$

so that the importance weight (function) is given by: 

$$
w_{t}(\mathbf{s}_{t},n) = \frac{p(n, \mathbf{s}_t \mid \mathbf{v}_{1:t})}{q_t(n, \mathbf{s}_t \mid \mathbf{v}_{1:t})} \propto \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t )  w_{t-1}^{n} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n}) }{\lambda_{t}^{n} q_t(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n}, \mathbf{v}_t)}
$$

Where I just emphasized the dependency of the weight on the hidden state $$\mathbf{s}_t$$ *and* the index $$n$$. This interpretation makes it very clear that in the implementation, *first we sample an index* $$n$$ with probability $$\lambda_{t}^{n}$$, *and then* we sample the particle from the corresponding transition density at particle $$n$$ (this is essentially what is known as ancestor sampling). The preweight is what enables the incorporation of information from the current measurement in the propagation of particles. You may wonder how all of this can be the same as what described in the previous section, where we derived the APF as a standard SMC algorithm with a different $$\gamma$$. In fact, and I think it's not immediate to see, if you substitute in the general SMC meta algorithm the specifics of the APF described in the previous section, you will find that at the end of iteration $$t$$ you will be resampling using a weight that includes information from $$\mathbf{v}_{t+1}$$. This is essentially the same as "delaying" the resampling step to the next iteration, and make use of that information to propagate the particles, instead of (e.g.) blindly use the transition density.

It is now also easy to see that if the proposal were equal to the transition density, and the preweight was $$w_{t-1}^{n} \color{green}{g}(\mathbf{v}_t \mid \boldsymbol{\mu}_{t}^{n}) $$, we would recover exactly the importance weight for APF derived in \eqref{eq30} with a different intepretation. It is also easy to see, now that we have talked about marginal particle filters, how we could get a marginalized version of the APF: simply marginalize over the indexes $$n$$ in the last equation, for both numerator and denominator, effectively performing inference for $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) $$.
This would turn the previous result into:

$$
w_{t}(\mathbf{s}_t) = \frac{\sum_{n=1}^{N} p(n,\mathbf{s}_t \mid \mathbf{v}_{1:t})}{\sum_{n=1}^{N}q_t(n,\mathbf{s}_t \mid \mathbf{v}_{1:t})} = \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_t) \sum_{n=1}^{N} w_{t-1}^{n} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{n})}{\sum_{n=1}^{N} \lambda_{t}^{n} q_t(\mathbf{s}_t \mid \mathbf{v}_t , \mathbf{s}_{t-1}^{n}) }
$$

This gives rise to what the authors of [5] call "Auxiliary Marginal Particle Filter". I believe the authors of APF wanted to keep the $$\mathcal{O}(N)$$ complexity of standard particle filtering, and this is why they did not do this last "marginalization" step. 

The topic explored in the next section is intimately connected with marginal filters, as previously hinted. 


## The Multiple Importance Sampling interpretation of particle filtering <a name="mis"></a>

Recently in [3] a novel re-intepretation of classic particle filters such as BPF and APF was published. This introduces a framework in which these filters emerge as special cases, and explains their properties under a Multiple Importance Sampling (MIS) perspective. MIS is a subfield of IS that is concerned with the use of multiple propoasals to approximate integrals and distributions. While this is similar to the Marginal Partice Filter, it more explicitly highlights the importance of the overlap of transition kernels and how this can be used to design a better filter.

Moreover, for the APF it assumed the common approximation to the predictive likelihood described earlier $$g(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t})$$, where $$ \boldsymbol{\mu}_{t} := \mathbb{E}_{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1})} [ \mathbf{s}_t ] $$. Finally, the proposal is selected to be the transition density. Since so far we have only talked about the APF as a "meta" algorithm, here is a concrete version (the most popular) where we choose the transition density to sample particles. It follows from either the first intepretation given, or from the original one.

<div id="example1">

<i> <b> Algorithm 2: APF </b> </i> <br> 

At time $t=1$: draw M i.i.d. samples from the prior proposal $ p(\mathbf{s}_1) $ <br>

At time $t \geq 2$, with particle/weight set $\left \{ \mathbf{s}_{t-1}^{m}, w_{t-1}^{m} \right \}_{m=1}^{M} $:

<ol>
  
<li> <b> Preweights computation </b> :
  <ul>
    <li> Compute $\boldsymbol{\mu}_{t}^{m} := \mathbb{E}_{ f(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}^{m})} [ \mathbf{s}_t ]$  for all $m$ </li>
    <li> Preweights are computed as:
    $ \lambda_{t}^{m} \propto {\color{green}g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) w_{t-1}^{m} $ for all $m$ </li>
  </ul> 
</li>
  
<li> <b> Delayed (multinomial) resampling step </b> : 
   Selecting resampled <i> indices </i> $ r^{m}, ~~ m= 1 \dots M$ with probability mass function given by $\Pr(r^{m} = j) = \lambda_{t}^{j}$ for $j \in \left \{ 1 \dots M \right \}$. Having this representation with resampled indices from the previous particle set instead of using a new particle set will be useful. 
 </li>
    
<li> <b> Propagation </b> : Sample $\mathbf{s}_{t}^{m} \sim {\color{blue}f}(\mathbf{s}_t \mid \mathbf{r}_{t-1}^{m}) $ or equivalently $\mathbf{s}_{t}^{m} \sim {\color{blue}f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{r^{m}}) $ for $m = 1, \dots, M$ </li>

<li> <b> Weight update </b> : Compute weights:
    $\tilde{w}_{t} =  \frac{g(\mathbf{v}_t \mid \mathbf{s}_{t}^{m})}{g(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{r^{m}} )}$ (as in eq. 30, without the multiplicative update since we always resample) </li> 
    
</ol>

</div>

<br>

We will frame this algorithm as a special case under the MIS intepretation of particle filtering. Recall that resampling (and propagating the resulting particles through a proposal) is equivalent to sampling from a mixture. In MIS, often the proposal is thought of as a weighted mixture of individual proposal. In this framework, explicit resampling + propagation is thus replaced with sampling from a single mixture proposal. We present here the MIS intepretation of BPF and APF, describing it below: 

___

**MIS intepretation of PFs**

At time $$t=1$$: draw M i.i.d. samples from the prior proposal $$ p(\mathbf{s}_1) $$ *and* set $$\color{#FF8000}{\lambda}_{1}^{m} = 1/M$$

At time $$t \geq 2$$: 

1. **Proposal adaptation/selection**: Select the Multiple Importance Sampling proposal of the form: 

$$
\color{#FF8000}{\Psi}_t(\mathbf{s}_t) = \sum_{m=1}^{M} \color{#FF8000}{\lambda}_{t}^{m} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{m})
$$

where $$\left \{ \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{m}) \right \}_{m=1}^{M} $$
are the transition densities or *kernels* centered at each of the previous particles.

Each kernel's weight $\lambda_{t}^{m}$ is computed as: 

$$ 

\color{#FF8000}{\lambda}_{t}^{m} = w_{t-1}  

$$  

*if the applied filter is BPF*

$$ \color{#FF8000}{\lambda}_{t}^{m} = \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) \cdot w_{t-1}^{m} $$ 

*if the applied filter is APF*

2. **Sampling**: Draw samples from the MIS proposal: $$\mathbf{s}_t \sim \color{#FF8000}{\Psi}_t(\mathbf{s}_t) $$

3. **Weighting**: Compute the normalized importance weights dividing target by proposal: 


$$\begin{equation}\begin{aligned}
w_{t}^{m} &\propto \frac{p(\mathbf{s}_{t}^{m} \mid \mathbf{v}_{1:t})}{\color{#FF8000}{\Psi}_t(\mathbf{s}_{t}^{m})} \\
&= \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) p(\mathbf{s}_{t}^{m} \mid \mathbf{v}_{1:t-1})}{\color{#FF8000}{\Psi}_t(\mathbf{s}_{t}^{m})} \\
&\approx \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) \sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}{\color{#FF8000}{\Psi}_t(\mathbf{s}_{t}^{m})} \\
&= \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) \sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}{\sum_{\color{red}{i}=1}^{M} \color{#FF8000}{\lambda}_{t}^{i} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})} \\
&\approx \frac{\color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) w_{t-1}^{m}}{\color{#FF8000}{\lambda}_{t}^{m}}

\end{aligned}\end{equation}\tag{31}\label{eq31}$$ 

___

In MIS, we are implicitly interested in the marginal $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) $$, and therefore we could say we are doing MPF. The chioce of mixture proposal is natural as it was for MPF: the numerator is a mixture, so it makes sense to try and match it with a mixture.  We will show how the algorithm, under certain approximations, leads to BPF and APF respectively with two different choices of $$\color{#FF8000}{\lambda}_t$$'s. We can also show how these choices are somewhat crude approximations: this will lead to the Improved Auxiliary Particle Filter.   

Let's start with the first of the three main stages, namely *proposal adaptation*. 
In this stage, weights akin to the APF "preweights" are computed in order to build the MIS proposal, which is a mixture (in this case of transition densities or *kernels*, but this is a choice really). We saw in the Marginal Particle Filter why it makes sense to have a mixture proposal: the numerator of $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) $$ is also a mixture, and moreover it makes sense to make it a mixture of the same kernels. So, we would like numerator and denominator to be close. A crucial fact is that both are a function of the latent state $$\mathbf{s}_t$$ , and we would like these two functions to be close in as wide a range of $$\mathbf{s}_t$$'s as possible. Of course, we don't know the value of the true $$\mathbf{s}_t$$.

Now, let's examine more closely how PFs act under the MIS interpretation. In \eqref{eq31} the last approximation is derived by essentially **assuming well separated kernels**. If the distance between kernels $$\left \{ \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{m}) \right \}_{m=1}^{M} $$ is high with respect to the kernel's width, then the two sums in the numerator and denominator can be well approximated by a single term. More precisely, consider that the $$m$$-th particle $$\mathbf{s}_{t}^{m}$$ has been simulated from kernel $$ \color{fuchsia}{k^{m}} \in \left \{ 1 \dots M \right \}$$, where the superscript $$m$$ emphasizes the dependency on $$m$$. If the other kernels $$ \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{j})$$ with $$j \neq \color{fuchsia}{k^{m}}$$ take small values when evaluated at $$\boldsymbol{\mu}_{t}^{\color{fuchsia}{k^{m}}}$$, then 

$$
\sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) \approx w_{t-1}^{\color{fuchsia}{k^{m}}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{fuchsia}{k^{m}}}) \qquad \sum_{\color{red}{i}=1}^{M} \color{#FF8000}{\lambda}_{t}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) \approx \color{#FF8000}{\lambda}_{t}^{\color{fuchsia}{k^{m}}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{fuchsia}{k^{m}}})
$$

which are indeed the approximations carried out in the last line of \eqref{eq31}.

The next issue to be discussed is the choice of mixture coefficients in the proposal $$  \color{#FF8000}{\lambda}_{t}^{m}$$. 
Notice that by setting the proposal to the (approximate) predictive distribution, $$\color{#FF8000}{\Psi}_t(\mathbf{s}_t) := p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) \approx \sum_{i=1}^{M} w_{t-1}^{i} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{i})  $$ , i.e. setting $$ \color{#FF8000}{\lambda}_{t}^{i} = w_{t-1}^{i} $$ then we are trying to match the right term of the numerator. This results in low discrepancy between denominator and numerator, *if* we had observations that are little informative, and recovers the BPF. It is also easily seen how plugging in the choice for the APF $$\lambda_t$$ indeed recovers the original APF. However, if the likelihood is high then this is clearly a bad choice. This can be seen by looking at teh numerator in \eqref{eq31} and plugging the likelihood into the sum that composes the predictive distribution (even if it does not depend on the sum):  $$\sum_{\color{red}{i}=1}^{M} \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) w_{t-1}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) $$ . Some kernels could be severely amplified with respect to others, thereby making the numerator very different to the proposal. The APF tries to improve on BPF *by trying to match the whole numerator with the proposal, and not just part of it*. It does so with a different choice of $$ \color{#FF8000}{\lambda}_{t}^{m} $$. The general idea is that to achieve what we want, kernels need to "communicate" in some way. Indeed, consider an edge case where all kernels have the same center. In this case, the APF $$\lambda$$ will amplify each transition kernel equally; consider what happens at the center of these kernels: APF amplifies kernel $$i$$ ignoring the fact that all of the other kernels $$j \neq i$$ will sample a lot from that part of the space too. The kernels are not communicating with each other. The IAPF achieves so by rescaling the predictive likelihood, evaluated at the center of the kernel, by a factor that depends on all other kernels:

$$\begin{equation}\begin{aligned}
\text{IAPF}~ \lambda_{t}^{m} \propto  \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m})\cdot \frac{\sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}{ \sum_{\color{red}{i}=1}^{M} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}} )} 
\end{aligned}\end{equation}\tag{32}\label{eq32}$$ 

Using this choice of $$\lambda_t$$ in the MIS meta-algorithm, *and* not performing the last approximation in \eqref{eq31} gives the *Improved Auxiliary Particle Filter*.

## The Improved Auxiliary Particle Filter <a name="iapf"></a>

The preweights of the IAPF, as we would have hoped, generalize the APF preweights in a natural way: when the kernels do not have significant overlap, we have:

$$\begin{equation}\begin{aligned}
 \lambda_{t}^{m} &\propto  \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m})\cdot \frac{\sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}{ \sum_{\color{red}{i}=1}^{M} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}} )}  \\
&\approx \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m})\cdot \frac{w_{t-1}^{m} \cancel{\color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{m})}}{ \cancel{\color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{m} )}} \\
&= \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m})\cdot w_{t-1}^{m}
\end{aligned}\end{equation}\tag{33}\label{eq33}$$ 

where the approximation comes from the assumption that $$ \color{blue}{f}(\mathbf{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{j}) \approx 0 $$ for $$ j \neq m$$. 

In summary, we have shown how the APF can be seen as a special case of the IAPF when kernels have negligible overlap. This assumption introduces simplifications both in the preweight $$\lambda_t$$, used to propagate the particles in areas with high likelihood, and in the importance weight that is used to build the empirical distribution of the target at $$t$$. 
Below, we used the same parameters as Elvira et al. [2] to reproduce the figure in the paper. It shows an example where the kernels have non-negligible overlap, and therefore the IAPF proposal matches the true posterior distribution better than the APF. Notice also that because of the informative likelihood, the BPF performs very poorly. A summary of the different choices of (unnormalized) preweights and importance weights for the different algorithms is also shown below.

$$
\begin{array}{c|lcr}
- & \text{BPF} & \text{APF} & \text{IAPF} \\
\hline
\color{#FF8000}{\lambda}_{t}^{m} & w_{t}^{m} & \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) \cdot w_{t-1}^{m} & \frac{\color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) \sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}{ \sum_{\color{red}{i}=1}^{M} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}} )} \\
w_{t}^{m} & \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m}) & \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m}) }{    \color{green}{g}( \mathbf{v}_t \mid \boldsymbol{\mu}_{t}^{r^{m}}) } & \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m}) \sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}( \mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) }{   \sum_{\color{red}{i}=1}^{M} \color{#FF8000}{\lambda}_{t}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}
\end{array}
$$


![iapf]({{ '/assets/images/iapf2.svg' | relative_url }})
*Fig. 5: Notice how the IAPF proposal best matches the posterior in this example. This is because the kernels have significant overlap, which is ignored by the preweights of APF. The kernels are plotted scaled by their importance weight; proposals and the true posterior integrate to 1. One quantitative measure to determine which proposal is better is the chi-squared distance to the true posterior, since it is proportional to the asymptotic variance of equation 23*




$$\require{color}$$
$$\require{bbox}$$




 
## References 
1. Elvira, V., Martino, L., Bugallo, M.F. and Djurić, P.M., 2018, September. In search for improved auxiliary particle filters. In 2018 26th European Signal Processing Conference (EUSIPCO) (pp. 1637-1641). IEEE.
2. Doucet, A. and Johansen, A.M., 2009. A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704), p.3.
3. Elvira, V., Martino, L., Bugallo, M.F. and Djuric, P.M., 2019. Elucidating the Auxiliary Particle Filter via Multiple Importance Sampling [Lecture Notes]. IEEE Signal Processing Magazine, 36(6), pp.145-152.
4. Naesseth, C.A., Lindsten, F. and Schön, T.B., 2019. Elements of Sequential Monte Carlo. Foundations and Trends® in Machine Learning, 12(3), pp.307-392.
5. Mike Klaas, Nando de Freitas, and Arnaud Doucet. Toward practical N2 Monte Carlo: the marginal particle filter.  In Proceedings of the Twenty-First Conference Annual Conference on Uncertainty in Artificial Intelligence (UAI-05), pages 308–315, Arlington, Virginia, 2005. AUAI Press.
6. Doucet, A., 1998. On sequential simulation-based methods for Bayesian filtering.
7. Doucet, A., Godsill, S. and Andrieu, C., 2000. On sequential Monte Carlo sampling methods for Bayesian filtering. Statistics and computing, 10(3), pp.197-208.
8. Godsill, S., 2019, May. Particle filtering: the first 25 years and beyond. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7760-7764). IEEE.
9. Arulampalam, M.S., Maskell, S., Gordon, N. and Clapp, T., 2002. A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. IEEE Transactions on signal processing, 50(2), pp.174-188.
10. Särkkä, S., 2013. Bayesian filtering and smoothing (Vol. 3). Cambridge University Press.
11. Li, T., Bolic, M. and Djuric, P.M., 2015. Resampling methods for particle filtering: classification, implementation, and strategies. IEEE Signal processing magazine, 32(3), pp.70-86.
12. Pitt, M.K. and Shephard, N., 1999. Filtering via simulation: Auxiliary particle filters. Journal of the American statistical association, 94(446), pp.590-599.




