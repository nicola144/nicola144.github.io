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
* Introduce approximate inference using importance sampling, in state space models
* Finally, describe the Auxiliary Particle Filter, its intepretation and the recent Improved Particle Filter by Elvira et al. [1]

The ideal target reader has familiarity with Bayesian inference, basics of Monte Carlo and importance sampling, basics of particle filters. However, if you are familiar with Bayesian inference in a "batch" setting (where data is processed all at once), you should be able to follow. If you are not, I will write a blogpost on Bayesian inference that assumes no prior background except basic rules of probability. Even then, I ambitiously hope that this post can be interesting to both Bayesian statistics experts who aren't aware of the work I will describe (these can skip to the sections on Auxiliary Particle Filters and Improved Auxiliary Particle Filters.) *and* people who see particle filters for the first time. 

1. [Brief introduction to sequential inference](#introduction)
    1. [General Bayesian Filtering](#generalfilter)
    2. [Recursive Formulations](#recursive)
2. [Particle Filtering](#pf)
    1. [Basics](#basics)
    2. [Choice of proposal and variance of importance weights](#isproposal)
    3. [Sequential Importance Sampling](#sis)
    4. [Resampling](#resampling)
3. [Propagating particles by incoporating the current measurement](#apf)
    1. [The effect of using the locally optimal proposal](#optimalproposal)
    2. [The Auxiliary Particle Filter](#apf2)
4. [The Multiple Importance Sampling Interpretation](#mis)
    1. [The Improved Auxiliary Particle Filter](#iapf)


## Brief introduction to sequential inference <a name="introduction"></a>

In Bayesian inference we want to update our beliefs on the state of some random variables, which could represent parameters of a parametric statistical model or represent some unobserved data generating process. Focussing on the "updating" perspective, the step to using Bayesian methods to represent dynamical systems is quite natural. The field of statistical signal processing has been using the rule of probabilities to model object tracking, navigation and even.. spread of infectious diseases. 
The probabilistic evolution of a dynamical system is often called a *state space model*. This is just an abstraction of how we think the state of the system evolves over time. Imagine we are tracking a robot's position (x,y coordinates) and bearing: these constitute a 3 element vector. At some specific timestep, we can have a belief, i.e. a probability distribution that represents how likely we think the robot is currently assuming a certain bearing etc. If we start with a prior, and define some likelihood function/ sampling process that we believe generates what we observe, we can update our belief over the system's state with the rules of probabilty.
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

It is reasonable to assume this: if we take a measurement, we don't expect its outcome to be dependent on previous states of the system, just the current one ($$\color{blue}{f}$$ and $$\color{green}{g}$$ seem arbitrary but they are common in the literature). For example, a classic Gaussian likelihood for would imply that the belief over $$\mathbf{v}_{t}$$ is a Normal with a mean being a linear combination of the state's coordinates.

This collection of random variables and densities defines the state space model completely. It is worth, if you see this for the first time, reflecting on the particular assumptions we are making. How the belief on $\mathbf{s}$ evolves with time could depend on many previous states; the measurement could depend on previous measurements, if we had a sensor that degrades over time, etc... I am not great at giving practical examples, but if you are reading this, you should be able to see that this can be generalized in several ways. 
Note that a lot of the structure comes from assuming some variables are (conditionally) independent from others. The field of probabilistic graphical models is dedicated to representing statistical independencies in the form of graphs (nodes and edges). One benefit of the graphical representation is that it makes immediately clear how flexible we could be. I am showing the graphical model for the described state space model in Figure 1, below. 


![hhm]({{ '/assets/images/tikz2.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1: Graphical model for the typical, first order Markov state space. Shaded nodes represent random variables whose value has been observed.*

In short, when the transition density and the observation densities are linear combination of their inputs with additive, i.i.d. Gaussian noise, then the state space model is often called a Linear Dynamical System (LDS). When variables are discrete, it is often called Hidden Markov Model (HMM). These are just labels. 

There are several tasks that we can perform on the state space model described above. Each of these has a fancy name, but you should of course recall that technically all we are doing is applying the sum and product rules. These tasks are associated to a *target* distribution which is the object of interest that we want to compute. Listing some of these: 

<p>
  <b>Filtering</b>: The target distributions are of the form: $p\left(\mathbf{s}_{t} | \mathbf{v}_{1: t}\right), \quad t=1, \ldots, T$. This represents what we have learnt about the system's state at time $t$, after observations up to $t$.<br>

  <b>Smoothing</b>:  The target distributions are of the form: $p\left(\mathbf{s}_{t} | \mathbf{v}_{1: T}\right), \quad t=1, \ldots, T$. This represent what we have learnt about the system's state after observing the *complete* sequence of measurements, and revised the previous beliefs obtained by filtering. <br>

<b>Parameter Estimation</b>: The target distributions are of the form: $p\left(\boldsymbol{\theta} | \mathbf{v}_{1: T}\right)= \int p\left(\mathbf{s}_{0: T}, \boldsymbol{\theta} | \mathbf{v}_{1: T}\right) \mathrm{d} \mathbf{s}_{0: T}$. The parameters $\boldsymbol{\theta}$ represent all the parameters of any parametric densities in the state space model. In the case that the transition and/or observation densities are parametric, and parameters are unknown, we can learn them from data by choosing those that both explain the observations well and also agree with our prior beliefs. Parameter estimation is sometimes referred to as *learning*, because parameters describe properties of sensors that can be estimated from data with machine learning methods. In other words, it is called learning just because it is cool. <br>
</p>



### General Bayesian Filtering <a name="generalfilter"></a>

#### Some notation
- The notation $$\mathbf{v}_{1:t}$$ means a collection of vectors $$ \left \{ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right \}$$
- Therefore, $$ p\left ( \mathbf{v}_{1:t} \right )$$ is a joint distribution: $$p\left ( \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_t \right ) $$
- Integrating $$ \int p(\mathbf{x}_{1:t}) \mathrm{d}\mathbf{x}_{i:j}$$ means $$ \underbrace{\int \dots \int}_{j-i+1} p(\mathbf{x}_{1:t}) \mathrm{d}\mathbf{x}_{i} \mathrm{d}\mathbf{x}_{i+1} \dots \mathrm{d}\mathbf{x}_{j} $$

In this post, I am only concerned with filtering, and will always assume that any parameters of <span style="color:blue">transition</span> or <span style="color:green">observation</span> densities are known in advance. 
Let's derive how to find the filtering distribution in the state space model described without many assumption on the densities.
Recall that the aim is to compute: $$ p\left(\mathbf{s}_{t} | \mathbf{v}_{1: t}\right)$$. Apply Bayes rule:  

$$
\require{cancel}
p\left(\mathbf{s}_{t} | \mathbf{v}_{1:t}\right) = \frac{ \overbrace{p \left( \mathbf{v}_{t} \mid \mathbf{s}_{t}, \cancel{\mathbf{v}_{1:t-1}} \right )}^{\mathbf{v}_t ~ \text{only dep. on} ~ \mathbf{s}_t} p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}_{1:t-1} \right )} = \frac{  \color{green}{g}\left( \mathbf{v}_{t} \mid \mathbf{s}_{t} \right ) p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right ) }{p\left( \mathbf{v}_t \mid \mathbf{v}_{1:t-1} \right )} \tag{3}\label{eq3}
$$

If this equation is confusing, think of the previous measurements $$\mathbf{v}_{1:t-1}$$ as just a "context", that is always on the conditioning side, a required "input" to all densities involved, with Bayes rule being applied to $$\mathbf{s}_{t}$$ and $$\mathbf{v}_{t}$$.
We know the current measurements only depends on the state, therefore $$p \left( \mathbf{v}_{t} \mid \mathbf{s}_{t}, \mathbf{v}_{1:t-1} \right ) = p \left( \mathbf{v}_{t} \mid \mathbf{s}_{t} \right ) = \color{green}{g}( \mathbf{v}_{t} \mid \mathbf{s}_{t} )$$, and only the right term in the numerator is left to compute. This term is a marginal of $$ \mathbf{s}_t$$, which means we have to integrate out anything else. If we were doing this very naively, each time we would integrate out all previous states, but by caching results a.k.a. Dynamic Programming, we only need to marginalize the previous state: 

$$
  p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right ) = \int p\left( \mathbf{s}_{t}, \mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1} \right ) \mathrm{d}\mathbf{s}_{t-1}
$$

Continuing, we split the joint with the product rule and exploit remember that the states are independent of previous measurements:

$$
\begin{equation}\begin{aligned}
  &= \int p\left( \mathbf{s}_{t} \mid  \mathbf{s}_{t-1}, \cancel{\mathbf{v}_{1:t-1}} \right ) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d}\mathbf{s}_{t-1} \\
  &= \int p\left( \mathbf{s}_{t} \mid  \mathbf{s}_{t-1} \right ) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d}\mathbf{s}_{t-1} \\
  &= \int \color{blue}{f}\left( \mathbf{s}_{t} \mid  \mathbf{s}_{t-1} \right ) p(\mathbf{s}_{t-1} \mid \mathbf{v}_{1:t-1}) \mathrm{d}\mathbf{s}_{t-1}
\end{aligned}\end{equation}\tag{4}\label{eq4}$$


And we are done, if you notice that the right side term in the integral is the filtering distribution at $$t-1$$, which we have already computed. 
In the literature names are given to the step that requires computing $$ p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right )$$ called *prediction*, because it's our belief on $$ \mathbf{s}_{t}$$ before observing the currrent measurement, and *correction* is the name given to the step $$ p\left(\mathbf{s}_{t} | \mathbf{v}_{1:t}\right) \propto \color{green}{g}\left( \mathbf{v}_{t} \mid \mathbf{s}_{t} \right ) \cdot p\left( \mathbf{s}_{t} \mid \mathbf{v}_{1:t-1} \right )$$, because we "correct" our prediction by taking into account the measurement. 

In a LDS, all computations have closed form solutions, and this algorithm instantiates into the *Kalman Filter*. For discrete  valued random variables, if the dimensionalities are small we can also do exact computations and the label this time is *Forward-Backward* algorithm for HMMs. 
When variables are non-Gaussian and/or transition/observation densities are nonlinear function of their inputs, we have to perform approximate inference. 
By far the most popular method is to use Monte Carlo approximations, and more specifically importance sampling. When we use importance sampling to approximate the filtering distribution, this is called *particle filtering*. 

### Recursive formulations <a name="recursive"></a>

Note that most particle filtering methods, which we will describe later, actually do not compute the filtering distribution with the correction and prediction steps explicitly. Instead, they express the filtering distribution by finding recursive relationships. Here, we show how this view is equivalent to the one just presented. The calculations will be instructive to understand particle filtering later. Consider the sequential estimation of a different distribution to the filtering, namely: 

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

Which is the indeed same result that we got through the prediction and correction steps. Again, this way of getting to the same result is useful for calculations involved in deriving particle filtering algorithms.  

<div id="example1">
  
$$\begin{equation}\begin{aligned}
 p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) =  p(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:t-1}) \frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} 
\end{aligned}\end{equation}\tag{9}\label{eq9}$$
</div>
<br>

<div id="example1">
$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t}) = \frac{p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{v}_{1:t-1})} 
\end{aligned}\end{equation}\tag{10}\label{eq10}$$
</div>
<br>

## Particle filtering <a name="pf"></a> 

### Basics <a name="basics"></a> 
Most things aren't linear and/or Gaussian, so we need approximate inference. Specifically in the filtering/sequential Bayes literature, importance sampling based methods are more popular than deterministic approximations such as Laplace's method, Variational Bayes and Expectation Propagation. 
Recall that the Monte Carlo method is a general tool to approximate integrals, expectations, probabilities with random samples: 

$$
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \mathrm{d}\mathbf{x} \approx \frac{1}{N} \sum_{n=1}^{N} f(\mathbf{x}_{n}) \qquad \mathbf{x}_n \sim p(\mathbf{x})
\tag{11}\label{eq11}$$

Where $$f(\mathbf{x})$$ is some generic function of $$\mathbf{x}$$. Monte Carlo approximations of this kind are very appealing since unbiased and consistent, and it is easy to show that the variance of the estimate is $$ \mathcal{O}(n^{-1})$$ *regardless* of the dimensionality of the vector $$\mathbf{x}$$. Another simple idea that we will use extensively in particle filtering is that these samples can not only be used to approximate integrals with respect to the target distribution $$p(\mathbf{x})$$, but also to approximate the target itself:

$$
p(\mathbf{x}) \approx \frac{1}{N}\sum_{n=1}^{N} \delta_{\mathbf{x}}(\mathbf{x}_{n})
$$

Where $$  \delta_{\mathbf{x}}(\mathbf{x}_{n})$$ is the Dirac delta mass evaluated at point $$\mathbf{x}_{n}$$. This is an empirical approximation of a density or "particle" approximation. It does not really represent the true density in some particularly meaningful way, but we only ever really need the density to compute moments. 
Often it is not possible to sample from the distribution of interest. Therefore we can use importance sampling, which is a technique based on the simple observation that we can sample from another, known distribution and assign a weight to the samples to represent their "importance" under the real target:

$$\begin{equation}\begin{aligned}
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] &= \int f(\mathbf{x}) \cdot p(\mathbf{x}) \mathrm{d}\mathbf{x} \\
&= \int \frac{f(\mathbf{x}) \cdot p(\mathbf{x})}{q(\mathbf{x})} \cdot q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \mathbb{E}_{q(\mathbf{x})} \left [ f(\mathbf{x}) \cdot \frac{p(\mathbf{x})}{q(\mathbf{x})} \right ]
\end{aligned}\end{equation}\tag{12}\label{eq12}$$

Under certain conditions, namely that $$ f(\mathbf{x}) \cdot p(\mathbf{x}) > 0 \Rightarrow q(\mathbf{x}) > 0$$, we have rewritten the expectation under a distribution of choice $$q(\mathbf{x})$$c alled *proposal* which we can sample from. Note that it is not possible to have $$ q(\mathbf{x}) = 0$$, as we will never sample any $$\mathbf{x}_{i}$$ from $$q$$ such that this holds. 
Let's return in the context of Bayesian inference, where we have a target posterior distribution $$ \pi(\mathbf{x}) = p(\mathbf{x} \mid \mathcal{D}) $$ where $$ \mathcal{D}$$ is any observed data. For example, in state space models $$\mathcal{D} = \mathbf{v}_{1:t}$$. Consider an integral of some function of $$ \mathbf{x}$$ under the posterior: 

$$\begin{equation}\begin{aligned}
\mathcal{I} = \mathbb{E}_{\pi(\mathbf{x})}[f(\mathbf{x})] = \int f(\mathbf{x}) \pi(\mathbf{x}) 
\end{aligned}\end{equation}\tag{13}\label{eq13}$$

Note that we can estimate this integral in two main ways with importance sampling: the former which assumes that we know the normalizing constant of the posterior $$ \pi(\mathbf{x})$$, and the latter estimates the normalizing constant too by importance sampling, with the same set of samples. Let's examine the latter option, called *self-normalized* IS estimator:

$$\begin{equation}\begin{aligned}
\mathbb{E}_{\pi} \left [ f(\mathbf{x}) \right ] &= \int f(\mathbf{x}) \pi(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \int  f(\mathbf{x})\frac{\pi(\mathbf{x})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{p(\mathcal{D})q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \frac{1}{p(\mathcal{D})}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \frac{1}{\int p(\mathbf{x}, \mathcal{D}) \mathrm{d} \mathbf{x}}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&=  \frac{1}{\int \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})}  q(\mathbf{x}) \mathrm{d} \mathbf{x}}\int  f(\mathbf{x})\frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} q(\mathbf{x}) \mathrm{d} \mathbf{x} \\
&= \frac{1}{\mathbb{E}_{q}\left [ \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} \right ]}
\cdot \mathbb{E}_{q}\left [ f(\mathbf{x}) \frac{p(\mathbf{x}, \mathcal{D})}{q(\mathbf{x})} \right ] \\
&\approx \frac{1}{\cancel{\frac{1}{N}}\sum_{n=1}^{N} \frac{p(\mathbf{x}_i , \mathcal{D})}{q(\mathbf{x}_i)}}
\cdot ~ \cancel{\frac{1}{N}} \sum_{n=1}^{N} f(\mathbf{x}_i) \frac{p(\mathbf{x}_i, \mathcal{D})}{q(\mathbf{x}_i)} := \widehat{\mathcal{I}}_{SN}
\end{aligned}\end{equation}\tag{14}\label{eq14}$$

The ratio $$ \frac{p(\mathbf{x}_i, \mathcal{D})}{q(\mathbf{x}_i)}$$ is called the importance weight: it can be intepreted as "adjusting" the estimate of the integral by taking into account that the samples were generated from the "wrong" distribution. If the normalizing constant was known, then we would build a *non-normalized* IS estimator that has different properties (with an almost equivalent derivation, omitted):  

$$

\widehat{\mathcal{I}}_{NN} := \frac{1}{N} \cdot \frac{1}{Z} \sum_{n=1}^{N}  f(\mathbf{x}_i) \frac{p(\mathbf{x}_i, \mathcal{D})}{q(\mathbf{x}_i)} 

$$

Where $$Z$$ is the normalizing constant of the posterior distribution $$\pi(\mathbf{x})$$. In this post we are only concerned with self-normalized estimators.

### Choice of proposal and variance of importance weights <a name="isproposal"></a>

It is pretty intuitive that our IS estimates can only be as good as our proposal. In general, we should seek a proposal that minimizes the variance of our estimators. This follows from the fact that the variance of a MC estimate (which is a sample average) is the expected square error from the true value of the integral. Let us see this by considering, for simplicity, the variance of the non-normalized estimator: 

$$ 

\mathbb{V}_{q} [ \widehat{\mathcal{I}}_{NN} ] = \mathbb{E} \left [ ( \widehat{\mathcal{I}}_{NN} - \mathbb{E} (  \frac{f(\mathbf{x}) \pi(\mathbf{x})}{q(\mathbf{x})} ] )^{2} \right ]

$$

### Sequential Importance Sampling <a name="sis"></a>

Let us now go back to the task of sequentially estimating a distribution of the form $$ \left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \}$$. This time however, we estimate any distribution by a set of weighted samples, a.k.a particles. 
First I am going to explain necessary notation. Note that the treatment in this section is very general and not specific to any particular state space model (hence not to the first order Markov one described earlier).  

* Let $$\gamma_{t}(\mathbf{s}_{1:t})$$ be the "target" distribution at time $$t$$ for states $$\mathbf{s}_{1:t}$$. Always keep track of all indices. For example, $$\gamma_{t}(\mathbf{s}_{1:t-1})$$ is a different object, namely $$\int \gamma_{t}(\mathbf{s}_{1:t}) \mathrm{d} \mathbf{s}_t $$. It is also different of course from $$\gamma_{t-1}(\mathbf{s}_{1:t})$$, which is just the target at $$t-1$$. Importantly, note that the usual "target" is **the unnormalized version** of whatever our distribution of interest is ($$ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ or $$p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t}) $$.

So, let's suppose then that we are trying to find a particle approximation for our target $$\gamma_{t}(\mathbf{s}_{1:t})$$. We can use importance sampling directly with a proposal distribution that also depends on $$\mathbf{s}_{1:t}$$ and find the  (unnormalized) importance weights: 

$$\begin{equation}\begin{aligned}
\tilde{w}_{t} = \frac{\gamma_t(\mathbf{s}_{1:t})}{\color{#FF8000}{q}_{t}(\mathbf{s}_{1:t})}
\end{aligned}\end{equation}\tag{15}\label{eq15}$$

With these , we can build the self-normalized importance sampling estimator as we have seen in the previous section. Recalling that all distributions we are dealing with are actually just a set of (possibly weighted particles), we can then approximate the (normalized) target by replacing the proposal with its particle approximation in \eqref{eq13}: 

$$\begin{equation}\begin{aligned}
 p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \approx \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n}) \qquad \mathbf{s}_{1:t}^{n} \sim \color{#FF8000}{q}_{t}(\mathbf{s}_{1:t})
\end{aligned}\end{equation}\tag{16}\label{eq16}$$

where $$w_{t}^{n}$$ are the normalized weights, and we are using $$N$$ sample trajectories for our proposal. Notice here the reason we can focus on $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ instead of $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t})$$ (more commonly needed). Since the latter is just a marginal of the former, and we have (approximate) samples from  $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$, we can form an approximation to $$p(\mathbf{s}_t \mid \mathbf{v}_{1:t})$$ simply as:

$$ 

p(\mathbf{s}_t \mid \mathbf{v}_{1:t}) \approx \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{t}}(\mathbf{s}_{t}^{n}) \qquad \mathbf{s}_{t}^{n} \sim \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})

$$

So, how is this different to non-sequential importance sampling? The problem is that without explicitly stating any assumptions/constraints on the proposal these calculations scale linearly with the dimension of the state space $$t$$. Let's see how it is possible to avoid this by simply imposing a simple autoregressive (time series jargon) structure on the proposal. 
Let our new proposal at time $$t$$ be the product of two factors: 

$$
q_{t}\left(\mathbf{s}_{1:t}\right)=q_{t-1}\left(\mathbf{s}_{1:t-1}\right) q_{t}\left(\mathbf{s}_{t} | \mathbf{s}_{1:t-1}\right)
$$

In other words, to obtain a sample from the full proposal at time $$t$$, we can just take the previous trajectory that was sampled up to $$t-1$$ and append a sample from $$ q_t\left(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}\right)$$. We can now exploit this proposal structure to express the weights at time $$t$$ as a product between the previous weights at $$t-1$$ with a factor. The algebraic trick uses multiplying and dividing by the target at $$t-1$$ to artificially obtain the weights at $$t-1$$:

$$\begin{equation}\begin{aligned}
 \tilde{w}_{t}\left(\mathbf{s}_{1:t}\right) &=\frac{\gamma_{t}\left(\mathbf{s}_{1:t}\right)}{\color{#FF8000}{q}_{t}\left(\mathbf{s}_{1:t}\right)} \\ &=\frac{1}{\color{#FF8000}{q}_{t-1}\left(\mathbf{s}_{1:t-1}\right)} \frac{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right)}{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right)} \frac{\gamma_{t}\left(\mathbf{s}_{1:t}\right)}{\color{#FF8000}{q}_{t}\left(\mathbf{s}_{t} | \mathbf{s}_{1:t-1}\right)} \\ &=\frac{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right)}{\color{#FF8000}{q}_{t-1}\left(\mathbf{s}_{1:t-1}\right)} \frac{\gamma_{t}\left(\mathbf{s}_{1:t}\right)}{\gamma_{t-1}\left(\mathbf{s}_{1:t-1}\right) \color{#FF8000}{q}_{t}\left(\mathbf{s}_{t} | \mathbf{s}_{1:t-1}\right)} \\
 &= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\gamma_{t}(\mathbf{s}_{1:t})}{\gamma_{t-1}(\mathbf{s}_{1:t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1})}
\end{aligned}\end{equation}\tag{17}\label{eq17}$$

Therefore, we can approximate our desired distribution as:

$$\begin{equation}\begin{aligned}
p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \approx \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n})
\end{aligned}\end{equation}\tag{18}\label{eq18}$$

with the weights $$w_{t}^{n}$$ defined as the normalized weights found in \eqref{eq15}.
This is the essence of SIS (Sequential Importance Sampling). Important note: this is a standard presentation you can find e.g. from Johansen et al [2]. However, you should note that for example, if we put this into context of state space models say, then the proposal can depend on measurements too. Crucially, although it would be natural to split it as: $$ \color{#FF8000}{q}_{t}\left(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}\right)= \color{#FF8000}{q}_{t-1}\left(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:\color{red}{t-1}}\right) \color{#FF8000}{q}_{t}\left(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}, \mathbf{v}_{1:\color{red}{t}}\right)$$ this is usually a *choice*, we could make both terms dependent on the current measurements! We will come back to this when discussing the Auxiliary Particle Filter. 
Ok, now it's time to apply SIS to the state space model we covered earlier. In this context, what we want is $$\left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \} $$ , hence our target $$\gamma$$ is the unnormalized posterior: $$\gamma_{t}(\mathbf{s}_{1:t}) := p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$. Recall that we can always get the filtering distribution from $$\left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \} $$. Now the recursion that we developed earlier in the post for the joint $$ p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$ becomes useful in deriving the weight update for SIS: 

$$\begin{equation}\begin{aligned}
\tilde{w}_t &= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\gamma_{t}(\mathbf{s}_{1:t})}{\gamma_{t-1}(\mathbf{s}_{1:t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1})} \\
&=  \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) \overbrace{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})}^{\cancel{\gamma_{t-1}(\mathbf{s}_{1:t-1})}}}{\cancel{\gamma_{t-1}(\mathbf{s}_{1:t-1})} \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})} \\
&= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})}
\end{aligned}\end{equation}\tag{19}\label{eq19}$$$$


If you are given a choice for the proposal $$\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}) $$, then you have a concrete algorithm to sequentially approximate $$\left \{ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t}) \right \}_{t \geq 1}$$, with constant time per update (remembering that throughout the algorithm only uses unnormalized weights, and only when one wants to approximate the desired distribution one has to normalize the weights). This algorithm is neat, but being a special case of IS it still suffers from variance of the weights scaling exponentially with time. This results in well known problems, the first of which is known under the names of *sample degeneracy* or *weight degeneracy*. Basically, if you actually run this after not-so-many iterations there will be one weight $$\approx 1$$ and all other will be zero, which equates to approximate the target with one sample. 

![hhm]({{ '/assets/images/sample-deg.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2: Sample or Weight degeneracy of SIS. The size of disks represent the size of the corresponding weight to a particle. Borrowed from Naesseth et al. [4]*


### Resampling <a name="resampling"></a> 

This is where Sequential Monte Carlo or Sequential Importance Resampling (SIR)/ particle filtering algorithms come into the picture. They mitigate the weight degeneracy issue by changing the particle set that was found in the previous iteration. They do so by resampling independently with replacement an equally sized particle set, where each sample is sampled with probability equal to its weight. This particular type of resampling is equivalent to sampling from a multinomial distribution with parameters equal to the weights, and is thus called multinomial resampling. Thus, we could see SMC/SIR as the same algorithm as SIS with an added step at the end of each iteration, where we resample particles according to their weights, and modify these to be $$1/N$$.


Let us put it into the same framework that we used to derive the weight update for SIS, and discuss some details. In vanilla SIS, at each iteration we build an approximate empirical distribution to our target, say $$ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$, that is represented by a weighted sum or "mixture" : $$ \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n})$$. The samples $$\left \{ \mathbf{s}_{1:t}^{n} \right \}_{n=1}^{N}$$ used in our approximation however come from the proposal $$ q_t(\mathbf{s}_{1:t})$$. We can obtain a set of samples approximately distributed according to $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ by multinomial resampling from our mixture approximation. Crucially, using these resampled particles, we can form a *different* estimator than $$ \sum_{n=1}^{N} w_{t}^{n} \delta_{\mathbf{s}_{1:t}}(\mathbf{s}_{1:t}^{n})$$, namely: $$ \frac{1}{N} \sum_{n=1}^{N}\delta_{\mathbf{r}_{1:t}} (\mathbf{r}_{1:t}^{n})$$. **This is because the samples actually come from** $$ p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$ (or whatever our target was) and thus to approximate the distribution itself we use its empirical approximation. The estimates (e.g. moments) under this approximation however have *higher variance* than the previous estimator. This is the price that we have to pay to mitigate the weight degeneracy: increase (temporarily) the variance of the estimator to however reduce the variance in the long run, many iterations later. 

--- 


**Algorithm 1: Sequential Monte Carlo / Sequential Importance Resampling *

At time $$t=1$$: 

1. **Propagation** : sample from proposal $$\mathbf{s}_{1}^{n} \sim \color{#FF8000}{q}_{1}(\mathbf{s}_1)$$
2. **Update**: compute weights $$w_{1}^{n} \propto \frac{p(\mathbf{s}_{1}^{n}, \mathbf{v}_{1})}{\color{#FF8000}{q}_{1}(\mathbf{s}_{1}^{n})}$$
3. **Resample**: $$\left \{ \mathbf{s}_{1}^{n} , w_{1}^{n} \right \}_{n=1}^{N} $$ to obtain $$ \left \{ \mathbf{r}_{1}^{n}, 1/N \right \}_{n=1}^{N} $$

At time $$t \geq 2$$:

1. **Propagation** : sample from proposal $$\mathbf{s}_{t}^{n} \sim \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{r}_{1:t-1}^{n})$$ and set $$ \mathbf{s}_{1:t}^{n} \leftarrow (\mathbf{r}_{1:t-1}^{n}, \mathbf{s}_{t}^{n})$$
2. **Update**: compute weights $$w_{t}^{n} \propto \frac{p(\mathbf{s}_{1:t}^{n}, \mathbf{v}_{1:t})}{\color{#FF8000}{q}_{t}(\mathbf{s}_{t}^{n} \mid \mathbf{r}_{1:t-1}^{n})}$$
3. **Resample**: $$\left \{ \mathbf{s}_{1:t}^{n} , w_{t}^{n} \right \}_{n=1}^{N} $$ to obtain $$ \left \{ \mathbf{r}_{1:t}^{n}, 1/N \right \}_{n=1}^{N} $$ 


---

HERE SAY WHY NOT MULTIPICATIVE UPDATE.
Finally, note that if we chose a proposal equal to the transition density, so $$ \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1}) = \color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})$$ , then the weight update simpli


![pf]({{ '/assets/images/bpff.svg' | relative_url }})
{: style="width: 100%;" class="center; height: 100%"}
*Fig. 3: Illustration of BPF for a one dimensional set of particles. Tikz figure with minor modifications from an original made by Víctor Elvira.*

![hhm]({{ '/assets/images/path-degg.svg' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4: Path degeneracy illustration in SMC/SIR/PF. Borrowed from Naesseth et al. [4]*


## Propagating particles by incorporating the current measurement <a name="apf"></a>

Let's talk about one particularly popular variation on the BPF. I will put it into the context of a generic SMC algorithm (for state space models), as did for the BPF, and explain different intepretations. I will start with the interpretation given by Johansen et al [2].

One of the motivations for APF is as follows. One can show that, if we had acces to the locally optimal proposal in SMC, then the weight update becomes an expression that does not involve the current state $$\mathbf{s}_t$$ at all. Notice that previously we have been performing propagation first, then weight update and resampling. Now if the weight update does not depend on the current state, then we could perform resampling before propagation. As Johansen et al [2] point out,  this yields a better approximation of the distribution as it provides a greater number of distinct particles to approximate the target. 

This interchange of propagation and resampling can be seen as a way of incorporating the effects of the current measurements $$\mathbf{v}_t$$ on the generation of states at time $$t$$, or of "filtering out" particles with low importance. 
Because in general we don't have access to the optimal proposal and thus can't do this exactly, we can try to "mimic" it, and this is what APF attempts to do. 

Before getting into APF however, let's actually inspect more explicitly what would happen if we used the locally optimal proposal in filtering. 

### The effect of using the locally optimal proposal <a name="optimalproposal"></a>

$$ 
\tilde{w}_{t} = \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\color{blue}{f}(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\frac{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_t)  }{ p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})} }
$$


$$ 
= \tilde{w}_{t-1} p(\mathbf{v}_t \mid \mathbf{s}_{t-1})
$$

The two main difficulties that using this proposal presents are:
1. Sampling from it can be hard  
2. Requires evaluation of $$ p(\mathbf{v}_t \mid \mathbf{s}_{t-1}) = \int \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}) \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}) \mathrm{d} \mathbf{s}_t$$

### The Auxiliary Particle Filter <a name="apf2"></a>

One interpretation of the APF is that of a standard SMC algorithm where the target that is propagated through each iteration is *not* the unnormalized filtering distribution, but rather $$\gamma_t(\mathbf{s}_{1:t}) = p(\mathbf{s}_{1:t}, \mathbf{v}_{1:\color{red}{t+1}}) $$. This is how it achieves the incorporation of the next measurements before propagation.  We will discuss a different derivation/intepretation of the APF in the next Section. 

The target, in the APF, therefore can be easily decomposed as:

$$\begin{equation}\begin{aligned}
\gamma_{t}(\mathbf{s}_{1:t}) := p(\mathbf{s}_{1:t} , \mathbf{v}_{\color{red}{t+1}}) &= \int p(\mathbf{s}_{1:t+1} , \mathbf{v}_{1:t+1}) \mathrm{d} \mathbf{s}_{t+1} \\
&= \int p(\mathbf{s}_{1:t} , \mathbf{v}_{1:t}) \cdot {\color{blue}{f}}(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}) \cdot {\color{green}{g}}(\mathbf{v}_{t+1} \mid \mathbf{s}_{t+1}) \mathrm{d} \mathbf{s}_{t+1} \\
&=  p(\mathbf{s}_{1:t} , \mathbf{v}_{1:t}) \int {\color{blue}{f}}(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}) \cdot {\color{green}{g}}(\mathbf{v}_{t+1} \mid \mathbf{s}_{t+1}) \mathrm{d} \mathbf{s}_{t+1} \\
&=  p(\mathbf{s}_{1:t} , \mathbf{v}_{1:t}) \cdot \underbrace{p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})}_{"predictive~likelihood"} 
\end{aligned}\end{equation}\tag{20}\label{eq20}$$

Which we see is equivalent to the product between what would be the target in standard SMC times the so called "predictive likelihood" $$p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})$$. The weight update can be derived by making use of this: 

$$\begin{equation}\begin{aligned}
\tilde{w} &= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\gamma_{t}(\mathbf{s}_{1:t})}{\gamma_{t-1}(\mathbf{s}_{1:t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t}\mid \mathbf{s}_{1:t-1})} \\
&=  \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot  \frac{p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t}) p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})}{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})} \\
&= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})} \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) p(\mathbf{v}_{t+1} \mid \mathbf{s}_{t})}{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})} p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1})} 
\end{aligned}\end{equation}\tag{21}\label{eq21}$$


Suppose we have been running the APF for $$1\dots t-1$$ iterations, and now we want a particle approximation of $$p(\mathbf{s}_{1:t} \mid \mathbf{v}_{1:t})$$. We can't compute the weights as in APF, because recall that it sequentially estimates something different to what we want, namely $$p(\mathbf{s}_{1:t-1} \mid \mathbf{v}_{1:t}) \cdot p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})$$. Thus, to compute the weights for our approximation, we have to use $$\gamma_t(\mathbf{s}_{1:t}) = p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})$$ and $$\gamma_{t-1}(\mathbf{s}_{1:t-1}) = p(\mathbf{s}_{1:t-1} , \mathbf{v}_{1:t}) \cdot p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})$$. Doing the whole derivation: 



$$\begin{equation}\begin{aligned}
\tilde{w}_t &= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot  \frac{p(\mathbf{s}_{1:t}, \mathbf{v}_{1:t})}{ p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})}  \\
&=  \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})}f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) g(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\cancel{p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1})} p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})} \\
&= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) g(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})}
\end{aligned}\end{equation}\tag{22}\label{eq22}$$ 


Note that in practice the predictive likelihood involves an intractable integral, so we have to approximate it with $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) $$. However, in the ideal case, selecting $$\color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{s}_{1:t-1}) =  p(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})$$ and $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) = p(\mathbf{v}_{t} \mid \mathbf{s}_{t-1})$$ leads to the so called "perfect adaptation" . 

Setting the approximation to the predictive likelihood to $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) = \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}(\mathbf{s}_{t})) $$ where $$ \boldsymbol{\mu}(\mathbf{s}_{t})$$ is some likely value is common. For example , if we choose as approximation to the predictive likelihood : $$\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) = \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}) $$  where $$\boldsymbol{\mu}_{t}$$ is the mean of $$ f(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) $$ *and* we also choose $$ \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1}) = f(\mathbf{s}_{t} \mid \mathbf{s}_{t-1})$$ , then we recover as special case the popular version of the APF weights: 

$$\begin{equation}\begin{aligned}
\tilde{w} &= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\hat{p}(\mathbf{v}_{t} \mid \mathbf{s}_{t-1}) \color{#FF8000}{q}_{t}(\mathbf{s}_{t} \mid \mathbf{v}_{t}, \mathbf{s}_{t-1})} \\
&= \tilde{w}_{t-1}(\mathbf{s}_{1:t-1}) \cdot \frac{\cancel{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})} \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})}{\color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}) \cancel{f(\mathbf{s}_{t}\mid \mathbf{s}_{t-1})}}
\end{aligned}\end{equation}\tag{23}\label{eq23}$$ 

## The Multiple Importance Sampling interpretation <a name="mis"></a>

Recently in [3] a novel re-intepretation of classic particle filters such as BPF and APF was published. This introduces a framework in which these filters emerge as special cases, and explains their properties under a Multiple Importance Sampling (MIS) perspective. MIS is a subfield of IS that is concerned with the use of multiple propoasals to approximate integrals and distributions, rather than just one. 

In this section, we drop the $$\gamma$$ notation since our target is always assumed to be $$p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t})$$. Note that in order to sequentially estimate this distribution and derive the importance weights, we will make use of \eqref{eq8} rather than \eqref{eq7}, that is our numerator of the (unnormalized) weights will be $$p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}) $$ and not $$p(\mathbf{s}_{1:t-1}, \mathbf{v}_{1:t-1}) \color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}) \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t})$$. 
Moreover, for the APF it assumed the common approximation to the predictive likelihood described earlier $$g(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t})$$, where $$ \boldsymbol{\mu}_{t} := \mathbb{E}_{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1})} [ \mathbf{s}_t ] $$. Finally, the proposal is selected to be the transition density. 

---

**Algorithm 3: APF (again)**

At time $$t=1$$: draw M i.i.d. samples from the prior proposal $$ p(\mathbf{s}_1) $$

At time $$t \geq 2$$, with particle/weight set $$\left \{ \mathbf{s}_{t-1}^{m}, w_{t-1}^{m} \right \}_{m=1}^{M} $$:

1. **Preweights computation**: 
    - Compute $$\boldsymbol{\mu}_{t}^{m} := \mathbb{E}_{\color{blue}{f}(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}^{m})} [ \mathbf{s}_t ]$$  for all $$m$$
    - *Preweights* are computed as:
    $$ \lambda_{t}^{m} \propto \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) w_{t-1}^{m} $$ for all $$m$$
2. **Delayed (multinomial) resampling step:** Sample with replacement from the previous particle set with probabilities $$\lambda_{t}^{m}$$ to obtain $$\left \{ \mathbf{r}_{t-1}^{m} \right \}_{m=1}^{M} $$ as well as associated means $$\left \{  ^{r \hspace{-1pt}}\boldsymbol{\mu}_{t}^{m} \right \}_{m=1}^{M} $$. Here however, instead of considering this generic resampling with a new particle set, let's be more specific. Notice that if this step uses multinomial resampling, what we just said is equivalent to: 
    - Selecting resampled **indices** $$ r^{m}, ~~ m= 1 \dots M$$ with probability mass function given by $$\Pr(r^{m} = j) = \lambda_{t}^{j}$$ for $$j \in \left \{ 1 \dots M \right \}$$. Having this representation with resampled indices from the previous particle set instead of using a new particle set will be useful. 
    
3. **Propagation**: Sample $$\mathbf{s}_{t}^{m} \sim \color{blue}{f}(\mathbf{s}_t \mid \mathbf{r}_{t-1}^{m}) $$ or equivalently $$\mathbf{s}_{t}^{m} \sim \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{r^{m}}) $$ for $$m = 1, \dots, M$$

4. **Weight update**: Compute weights:
    $$\tilde{w}_{t} =  \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m})}{\color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{r^{m}} )}$$

---

It was not immediate for me to check that indeed this algorithm does exactly the same thing as if we used Algorithm 2 with the assumptions above.

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

\end{aligned}\end{equation}\tag{24}\label{eq24}$$ 

___

We will show how the algorithm, under certain approximations, leads to BPF and APF respectively with two different choices of $$\lambda$$'s. We can also show how these choices are somewhat crude approximations: this will lead to the Improved Auxiliary Particle Filter.   

Let's start with the first of the three main stages, namely *proposal adaptation*. 
In this stage, weights akin to the APF "preweights" are computed in order to build the MIS proposal, which is a mixture (in this case of transition densities or *kernels*, but this is a choice really). The accuracy of inferences depends on the discrepancy between the numerator and denominator in \eqref{eq24}. 

Let's examine more closely again how PFs act under the MIS interpretation. In \eqref{eq24} the last approximation is derived by essentially **assuming well separated kernels**. Pay attention to the fact that \eqref{eq24} is a function of a specific particle $$m$$. If the distance between kernels $$\left \{ \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{m}) \right \}_{m=1}^{M} $$ is high with respect to the kernel's width, then the two sums in the numerator and denominator can be well approximated by a single term. More precisely, consider that the $$m$$-th particle $$\mathbf{s}_{t}^{m}$$ has been simulated from kernel $$ \color{fuchsia}{k^{m}} \in \left \{ 1 \dots M \right \}$$, where the superscript $$m$$ emphasizes the dependency on $$m$$. If the other kernels $$ \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{j})$$ with $$j \neq \color{fuchsia}{k^{m}}$$ take small values when evaluated at $$\boldsymbol{\mu}_{t}^{\color{fuchsia}{k^{m}}}$$, then 

$$
\sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) \approx w_{t-1}^{\color{fuchsia}{k^{m}}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{fuchsia}{k^{m}}}) \qquad \sum_{\color{red}{i}=1}^{M} \color{#FF8000}{\lambda}_{t}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) \approx \color{#FF8000}{\lambda}_{t}^{\color{fuchsia}{k^{m}}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{fuchsia}{k^{m}}})
$$

which are indeed the approximations carried out in the last line of \eqref{eq24}.

The next issue to be discussed is the choice of mixture coefficients in the proposal $$  \color{#FF8000}{\lambda}_{t}^{m}$$. 
Notice that by setting the proposal to the (approximate) predictive distribution, $$\color{#FF8000}{\Psi}_t(\mathbf{s}_t) := p(\mathbf{s}_{t} \mid \mathbf{v}_{1:t-1}) \approx \sum_{i=1}^{M} w_{t-1}^{i} \color{blue}{f}(\mathbf{s}_t \mid \mathbf{s}_{t-1}^{i})  $$ , i.e. setting $$ \color{#FF8000}{\lambda}_{t}^{m} = w_{t-1}^{m} $$ then we are trying to match the right term of the numerator. This results in low discrepancy between denominator and numerator, *if* we had observations that are little informative, and recovers the BPF. However, if the likelihood is high then this is clearly a bad choice. This can be seen by plugging the likelihood into the sum that composes the predictive distribution:  $$\sum_{\color{red}{i}=1}^{M} \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) w_{t-1}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) $$ . Some kernels could be severely amplified with respect to others, thereby making the numerator very different to the proposal. The APF tries to improve on BPF *by trying to match the whole numerator with the proposal, and not just part of it*. It does so with a different choice of $$ \color{#FF8000}{\lambda}_{t}^{m} $$ in a very natural way: the only term we were not matching previously is $$ \color{green}{g}(\mathbf{v}_{t} \mid \mathbf{s}_{t}^{m}) $$. Importantly, this is a function of $$\mathbf{s}_{t}^{m}$$


## The Improved Auxiliary Particle Filter <a name="iapf"></a>


$$
\begin{array}{c|lcr}
- & \text{BPF} & \text{APF} & \text{IAPF} \\
\hline
\color{#FF8000}{\lambda}_{t}^{m} & w_{t}^{m} & \color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) \cdot w_{t-1}^{m} & \frac{\color{green}{g}(\mathbf{v}_{t} \mid \boldsymbol{\mu}_{t}^{m}) \sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}{ \sum_{\color{red}{i}=1}^{M} \color{blue}{f}(\boldsymbol{\mu}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}} )} \\
w_{t}^{m} & \color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m}) & \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m}) }{    \color{green}{g}( \mathbf{v}_t \mid \boldsymbol{\mu}_{t}^{r^{m}}) } & \frac{\color{green}{g}(\mathbf{v}_t \mid \mathbf{s}_{t}^{m}) \sum_{\color{red}{i}=1}^{M} w_{t-1}^{\color{red}{i}} \color{blue}{f}( \mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}}) }{   \sum_{\color{red}{i}=1}^{M} \color{#FF8000}{\lambda}_{t}^{\color{red}{i}} \color{blue}{f}(\mathbf{s}_{t}^{m} \mid \mathbf{s}_{t-1}^{\color{red}{i}})}
\end{array}
$$








$$\require{color}$$
$$\require{bbox}$$




 
## References 
1. Elvira, V., Martino, L., Bugallo, M.F. and Djurić, P.M., 2018, September. In search for improved auxiliary particle filters. In 2018 26th European Signal Processing Conference (EUSIPCO) (pp. 1637-1641). IEEE.
2. Doucet, A. and Johansen, A.M., 2009. A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704), p.3.
3. Elvira, V., Martino, L., Bugallo, M.F. and Djuric, P.M., 2019. Elucidating the Auxiliary Particle Filter via Multiple Importance Sampling [Lecture Notes]. IEEE Signal Processing Magazine, 36(6), pp.145-152.
4. Naesseth, C.A., Lindsten, F. and Schön, T.B., 2019. Elements of Sequential Monte Carlo. Foundations and Trends® in Machine Learning, 12(3), pp.307-392.
5. Särkkä, S., 2013. Bayesian filtering and smoothing (Vol. 3). Cambridge University Press.



