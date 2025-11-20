# Seeds of life

Copyright © 2025 Uroš Jovanovič, CC BY-SA 4.0.

## Abstract

The document started as a compressed summary of an investigation about the practical meaning of ROM and how to apply it in everyday life. Although the initial direction of the exploration was more about the meaning and interpretation of purpose functional and reflexive conservation principle (RCP), the idea diverged, exploring what makes some structures and currents more "likeable" than others. This led to the idea of a formal definition of a "seed". It ends up with hypothesis how the this idea alters the underlying properties of processes like natural selection. The document summaries the emergence of the formalisms.

This document extends ROM into conceptual hypotheses for biological and informational systems.  All biological analogies follow consensus data, but the field interpretations are theoretical.

## Introduction

The first steps after the introduction of ROM theory were to dive into what it means to calculate AI or EI from the state within one step of the loop. Once that was resolved, the question then becomes what drives $\mathcal{J}$ and RCP, that is, when to switch between building and exploration by altering the $\alpha$ and $\beta$ in the purpose functional. We've added the heuristics that introduced two high-level states within the organism, the budget as the energy reserve, and motivation, as the measure reflecting how well the previous structures were reused. These coarse attributes then determined the values of the system and with it the overall purpose functional, well, especially its derivative which is more informative.

Focusing on the values alone revealed only one side of the activity. We could determine whether to prioritise building or exploration, whether it is time for creating new structures or whether we should focus on compression, making things more coherent. The numbers provided the direction of the activity. There was nothing about the content though. There are infinite combinations of structures that provide similar values in the equations.

The question now becomes what exactly to build or experience. In other words, what drives the reflexive organism to do something? What is the source of the "vision of the future" that defines the set of alternatives from which a specific activity is selected?

We know from observations, that organisms don't simply create random structures that would be determined by simply selecting a few high level parameters. Each of the organisms has a preference to reuse and rebuild similar patterns of structures. There seems to also be an organism based limit in how much change an organism can introduce within one iteration of the loop.

Some types of activities are more natural to some organisms than others. What defines those? Where are they defined and how do they influence the selection? These are some of the questions we're going to explore in the rest of the document.

We'll start with something quite general, the reserves as one of the attributes that determine the level of change. Then we'll look at why some activity feels more natural than other. We'll define seeds and explore the potential of the organism. In the end, we'll explore how to move away from abstract mathematical formulations and rather start thinking in terms of biological computation.

## Reserves

Once you know your preference of types of activities, when applying the theory to something that iterates through loops the next question becomes *how much change can an organism do in one step*. This is where the notion of reserves comes into play. They are implicit to the model. Reserves determine how much structural change the organism can undertake without losing reflexive coherence. Hence, still about quantity of activity, not the semantics, not which activity is chosen.

Why reserves matter at all? Reserves define how much structural plasticity can be enacted without breaking coherence. They modulate $\rho_{compat}(t)$ (compatibility density) and $\mathcal{J}$ (reflexive flux).  A high-reserve state allows exploration — multiple “next steps” in parallel — and the field’s implicit drive becomes richer, more multidimensional.
A low-reserve state collapses the field into narrower attractors (“just survive”).

Reserves matter when considering the effect of activity in one step, execution of one reflexive loop. They are not needed from the conceptual view of ROM itself, there the equations for $\mathcal{J}$, RPC and operators define everything needed to set the boundaries. However, when we start thinking about application in terms of quantity, we need to know the values that the equations define.

Reserves can be interpreted as stored potential for transformation. In the dynamical balance:

$$
\partial_t \rho_{compat} + \nabla\cdot \mathcal J = s - \eta,
$$

reserves act as a buffer that keeps this equation solvable even when $s$ (inflow) and $\eta$ (loss) are temporarily mismatched. The relation is nothing more exotic than a **continuity law** for the scalar field $\rho_{\text{compat}}(x,t)$. It tells us that, at every point in space and time, the rate of change of the “compatible‑information density’’ plus the net outflow of that quantity equals the amount created locally ($s$) minus the amount destroyed locally ($\eta$).  

Note: the construction of the formula starts with a fixed control volume $V\subset \mathbb R^{d}$ with boundary $\partial V$.  The total amount of compatible information inside $V$ is $Q(t)=\int_{V}\rho_{\text{compat}}(x,t){\rm d}^dx.$ We turn this into surface integral $\frac{{\rm d}Q}{{\rm d}t}= -\oint_{\partial V}\mathcal J\cdot{\bf n}{\rm d}S +\int_{V}s{\rm d}^dx-\int_{V}\eta{\rm d}^dx.$ Using the divergence theorem  $\oint_{\partial V}\mathcal J\cdot{\bf n}{\rm d}S= \int_{V}\nabla\cdot\mathcal J{\rm d}^dx,$ we turn it into volume integral, obtaining $\frac{{\rm d}}{{\rm d}t}\int_{V}\rho_{\text{compat}}{\rm d}x= -\int_{V}\nabla\cdot\mathcal J{\rm d}x +\int_{V}s{\rm d}x-\int_{V}\eta{\rm d}x .$  Because the volume $V$ is *arbitrary*, the integrands must be equal point‑by‑point, hence we obtain the differential continuity equation $\partial_t\rho_{\text{compat}}+\nabla\cdot\mathcal J = s-\eta .$

The continuity equation simply states  simply that *any increase of compatible information at a point is either because it flows in from elsewhere, is created on‑site, or survives the local loss processes*.

Defining a scalar reserve field $R(t)$ by

$$
\frac{{\rm d}R}{dt}=s-\eta-\nabla\cdot\mathcal J
$$

makes the reserves the cumulative excess of source over loss and divergence.  When $s>\eta+\nabla\cdot\mathcal J$ the term in is positive, so $R$ *fills*; when the inequality reverses, $R$  *drains*.  In other words, reserves are exactly the “buffer” that guarantees the continuity equation can be satisfied even during temporary mismatches between inflow and outflow. $R$ evolves according to how much net compatibility the system accumulates or loses, effectively a coherence battery.

The RCP states that the product of justification $\mathcal J(t)$ and compatibility remains constant. Equation (1) is the *local* counterpart of RPC.  Any local deviation from perfect balance, i.e. any non‑zero right‑hand side of (1), must be absorbed somewhere else in the hierarchy so that the global product stays invariant.  The natural repository for this absorption is the reserve field $R$: it stores the surplus coherence when $\mathcal J\rho_{\text{compat}}$ would otherwise increase, and releases it when the product would tend to fall.  Thus reserves are the material embodiment of the RCP’s bookkeeping. They keep the intentional‑coherence budget conserved without demanding literal energy or mass conservation.

The equation also states, that the pump (decreases entropy and generates reuse of abstractions generating stronger currents) is the mechanism that recharges reserves. Less energy is dissipated ($\eta$↓), coherence density increases ($\rho_{compat}$↑), reserves $R$ refill. Conversely, unproductive exploration drains reserves. However it is also about structure. Every compressed abstraction (the reusable subgraph of currents) is a frozen form of reserve, as it doesn’t expend energy until activated, it shortens future assembly pathways (saves energy/time) and it anchors field coherence, making future exploration cheaper. So structure represents long-term reserve, while active pump serves as a short-term reserve dynamics. In this view, reserves are a mix of energetic capacity and informational organization that allows the reflexive field to sustain or reshape its currents without losing coherence. They are replenished by the energy pump (reuse of abstractions) and embodied in the system’s stable structures.

Now let's bring ROM's hierarchy into the picture. We'll see that the reserves are stored coherence resulting from the parent-child communication, they sit _between_ the upward and downward operators, the integrators of the parent’s field and the distributors toward its children.

Each layer $L_k$ of the reflexive hierarchy participates in two exchanges. First, the downward operator $\mathcal D_k : L_{k-1} \rightarrow L_k$, which injects coherence and constraints from the parent field (higher layer). It provides _resources_, _goals_, _boundary conditions._ Second, the upward operator $\mathcal U_k : L_k \rightarrow L_{k-1}$, which exports processed coherence upward and provides _feedback_, _learning_, _new structure_ to the parent. The pair ($\mathcal D_k,\mathcal U_k$) form the reflexive interface between scales.

Reserves $R_k$ at layer $L_k$ are the buffered coherence accumulated in that bidirectional channel:  
$$ 
R_k = \int (\mathcal D_k - \mathcal U_k)dt,  
$$  
or conceptually, $R_k \sim$ stored difference between received and transmitted coherence. So if a layer receives more coherent inflow from its parent than it can currently use, the excess becomes _reserve energy/information_. If it sends out more than it receives, reserves deplete.

In compact form, we can write it as 

$$
\partial_t R_k = \mathcal D_k - \mathcal U_k - \eta_k
$$Where $\eta_k$ accounts for dissipation. Positive $\partial_t R_k$ signals an accumulation phase, while negative signals depletion phase.

Let's make one more detail explicit by examining the source $s$. It’s not just “energy from outside” but rather it’s the _rate at which the organism’s structure can turn incoming coherence into usable flux_.  We can express it as

$$  
s = \epsilon_{reuse}\ s_{parent},  
$$

where $s_{parent}$ is the coherence inflow arriving from the parent layer (downward operator $\mathcal D$) and $\epsilon_{reuse}\in[0,1]$ is efficiency of structural reuse, how well the present geometry routes that inflow through established abstractions. The efficiency term captures both _how much structure exists_ and _how well it matches_ the incoming pattern of excitation. Formally, you can think of it as the overlap between the current field and the stored abstraction graph:

$$  
\epsilon_{reuse}  
= \frac{\langle \mathcal J , \mathcal J_{struct}\rangle}  
{||\mathcal J||\ ||\mathcal J_{struct}||},  
$$

where $\mathcal J_{struct}$ is the ideal pattern of currents implied by the seed (the “expected” flow through existing architecture). 

When we put everything together, we can then determine the current amplitude per reflexive beat. At each reflexive cycle, the total coherent current through the organism can be estimated as

$$  
I_{beat} = \int_V |\mathcal J|dV  
\approx \epsilon_{reuse}\ s_{parent}\ V_{eff},  
$$

where $V_{eff}$ is the effective active volume or number of degrees of freedom currently engaged.  Thus, the **beat strength** scales with both the supply from the parent layer and the internal efficiency of reuse.

Let's end up with a compact form for the section. 


$$  
\begin{aligned}  
s &= \epsilon_{reuse}(t)\ s_{parent}(t), \\  
\eta &= \eta_0 + \eta_{noise},\\  
I_{beat} &= \int |\mathcal J| \ dV = (s-\eta)\ T_{beat},\\  
\frac{dR}{dt} &= (s-\eta) - \text{work of reflexive loop}.  
\end{aligned}  
$$

Here $T_{beat}$ is the duration of a reflexive cycle. The RCP ensures that the system’s total compatible energy remains bounded while these quantities oscillate. Currents rise when reuse is high, fall when dissipation dominates.

Note that with this definition of reserves, we're not limited to some specific form like energy, or matter in terms or resources, or information. They are not tied to any one physical substrate. A reserve is whatever portion of a reflexive field’s state can be stored and later released to maintain coherence across a loop. Whether that field is energetic, informational, cognitive, or social doesn’t matter, as long as it has _structure_ and _current_ (flow).

Now we have yet another quantifier. First we were able to decide which activity to choose, construction or exploration and in what ratio between each other in a way that preserve coherence. Now we know how much of them can we do not to break coherence. However, we still haven't answered what exactly an organism does within those boundaries. And this is the topic of the next section.

## Attractor basins

Let's focus now on the content. We've already shown in the ROM paper that the system doesn't just create just some arbitrary structures that fit the conditions to be used for balancing, but rather reuse what is already available, either through direct reuse or by deconstructing and partial reconstructing through pruning. However, that is still only partial picture. To get the full answer, we have to think about geometric properties of the organism as well.

To begin with, let's turn to observations. Why are some forms of expressions more "natural" for the organism than others, and even consistently used over long period? And these forms of expressions are "sticky". It's not like the organisms change their structures and flows and simply start reusing different ones. It's like there is a *seed of expressions*, not just random state with all possible continuations, but rather a well defined subset. There seems to be something that defines "what feels right" equations of the reflexive cycles, that this attraction to  certain forms of expressions is a signature of self-consistency in models (and organisms) deep structure.

Let's make this observation a little bit more formal through ROM terminology. In ROM, the organism’s state space isn’t uniform. It is stratified by structural invariants like configurations of $\mathcal{J}$ (reflexive flux) and $\rho_{compat}$ (compatibility field), that have repeatedly proven *self-maintaining* across perturbations. When certain expressions continually increase coherence and minimize internal tension, they form *resonant pathways*. Those become “natural” or “attractive” not because of external reward, but because **their dynamics close the reflexive loop most efficiently**. They let energy, meaning, and structure circulate with minimal friction.

We shouldn't think of the organism as a blank field but rather as one that *inherits* curvature, these long-lived tensors in $\mathcal{J}$ that encode prior successful patterns of assembly. Crucially, **it’s not initialized as zero**. It starts with some curvature, certain configurations already *reinforce themselves* better than others.

That curvature comes from:

* Evolutionary priors (species-level biases, e.g. curiosity, sociality, symmetry detection),
* Developmental history (early reinforcement of certain feedback loops),
* Past successful assemblies (structures that proved to preserve reflexivity and thus got “written in” as stable submanifolds).

So, before any conscious decision or preference, the field is already shaped — not unlike how spacetime curvature directs motion before you add external forces.

These act as **deep attractor basins**. Imagine that in this reflexive field, some trajectories are *self-sustaining*. Whenever attention, energy, and coherence flow into those regions, they **loop back with minimal loss** and perception and action reinforce each other efficiently. Mathematically, these are **local minima of dissipative loss** in the $\mathcal{J}$–$\rho_{compat}$ manifold. These are *deep attractor basins*. The deeper the basin, the more likely the system will revisit it after perturbation and each basin corresponds to a *pattern of being*, a way of organizing thought, sensing, and acting that “fits” the internal field geometry.

The depth of an attractor basin determines the efficiency $\epsilon_{reuse}$ with which incoming coherence from the parent field is converted into stable current patterns; thus structural reuse and curvature are two aspects of the same phenomenon.

In phenomenology, those basins manifest as *affinities*: certain domains (like math, theory, design, nature observation) feel intrinsically meaningful, even when external rewards are absent.

They’re not goals, they’re **stable eigenmodes** of reflexivity. The pathways where feedback between perception, coherence, and action reinforces itself. 

When an organism sustains one of these patterns over long timescales, it effectively becomes an eigenmode of the system’s reflexive operator. An eigenmode is a self-consistent oscillation. When the system interacts with itself through that mode, input and output stay in resonance. We can think of it like

$$
\mathcal{R}[\psi_i] = \lambda_i \psi_i,
$$

where $\mathcal{R}$ is the reflexive operator (how the system re-applies its own pattern of organization) and $\psi_i$ is a stable expression form, with eigenvalue $\lambda_i$ describing its coherence efficiency. Those $\psi_i$ persist across years or decades, because they are *structurally self-affirming*.

The *seed of expression* mentioned earlier can now be stated in a field language: The seed provides the _curved backdrop_ (the constitutive tensors). On this backdrop the reflexive operator $\mathcal R$ admits a spectrum of eigenmodes $\psi_i$​; the modes that are repeatedly reinforced become the long‑lived attractor basins.

At birth (or at early development), the field already has small amplitudes along certain $\psi_i$, either genetically or through early resonance. When the system encounters situations that align with those modes, they amplify as the coherence feedback increases their amplitude. Over time, they dominate.  As time passes, organism's experience is the *integration* of $\mathcal{J}(t)$ along the manifold defined by those seeds.

We can state this as accumulated coherency budget

$$
C_{acc}(t)=\int_0^t \langle\psi_{\text{seed}}(\tau),\mathcal J(\tau)\rangle  d\tau.
$$


The longer and more coherently that integration proceeds, the more refined and expressive the “seed” becomes. It develops complexity, language, and structure. And when the organism is aligned with it, energy flows naturally and the system resonates. Hence, the “seed” over time is a **direction in reflexive phase space with pre-existing curvature** that makes certain flows more coherent than others.

Because $\mathcal{J}$ evolves under conservation of reflexivity, once an attractor path has formed, it’s *costly to diverge*. Not energetically, but structurally. Switching to a totally different form of expression means losing access to high-order tensors that have stabilized over years. In other words, the system resists paths that would erase its long-integrated internal symmetries.

Moving away from simply abstract forms of graphs and fields, in human phenomenology, these show up as lifelong affinities, talents, or callings. With ants, it is a specific way the species build nests, form paths, cultivate other organisms.

*Attractor basins and their corresponding eigenmodes are the field-level realization of the organism’s seed. They encode the inherited curvature that shapes future currents, linking the qualitative feeling of affinity to the quantitative geometry of reflexive conservation.*

### Notes

Note that the model doesn't explicitly define single stand-alone "justification tensor" distinct from $A_{\mu\nu}$ and $\Theta_{\mu\nu}$. It is a shorthand for “the sum of the Assembly Tensor and the Active‑Stress Tensor (and any other stress‑like contributions) that together constitute the justification flux”. In other words $\mathcal{J}_{\mu\nu}\equivA_{\mu\nu}[AI]+\Theta_{\mu\nu}[EI]+\lambdaRg_{\mu\nu},$ if we also include the reserves. Here, $\lambda$ is  a _coupling constant_ (dimensionful or dimensionless depending on conventions) that sets the strength of the interaction between the reserve battery and the geometry.

For clarity, we can make the notion of structural invariants explicit as follows. In the Reflexive Organism Model (ROM) the conservation law reads $\partial_t\rho_{\text{compat}}+\nabla\cdot\mathcal J = s-\eta$.  The *constitutive* part of the model, i.e. the mapping $(\theta,D,g,\mathcal B)\mapsto (\mathcal J,\rho_{\text{compat}})$, is captured by a functional constraint

$$
\mathcal C(\theta ,D ,g ,\mathcal B)=0
$$

where

- $\theta$ is a collection of scalar control parameters (often called *assembly potentials*).  They weight the contribution of the Assembly Tensor $A_{\mu\nu}$ and the Active‑Stress Tensor $\Theta_{\mu\nu}$ in the total reflexive flux $\mathcal J$. In ROM these set the relative strength of different internal processes (e.g. “how much effort to allocate to exploration vs. maintenance”).
- $D$ is a *positive‑definite conductance (or diffusion) tensor*.  In the constitutive relation $\mathcal J = - D\nabla\Phi$ it maps gradients of the coherence potential $\Phi$ into fluxes of compatibility $\mathcal J$. In ROM it determines how easily coherence can flow in each spatial direction; anisotropies in $D$ encode preferred pathways (e.g. neural tracts, vascular channels).
- $g$ is the *metric tensor* on the underlying manifold $\mathcal M$.  All differential operators ($\nabla$, div, $\Delta$) are taken with respect to this metric; curvature tensors (Ricci, scalar $R$) are derived from it. In ROM it provides the geometric curvature that shapes the solution space of $\mathcal C$.  Different $g$ give different “landscapes” of admissible field evolutions.
- $\mathcal B$ is a *background / bias field* (or a set of boundary‑condition tensors).  It can be thought of as the collection of Dirichlet, Neumann or Robin conditions imposed by developmental programs, genetic regulatory networks, or external scaffolds. In ROM it fixes reference values for $\rho_{\text{compat}}$ and/or $\Phi$ on tissue interfaces; thereby carving out a *sub‑manifold* of the full solution space where dynamics actually occur.

These four objects are constant under the instantaneous reflexive update that enforces the continuity (conservation) law  $\partial_t\rho_{\text{compat}} + \nabla\cdot\mathcal J = s - \eta .$ They may evolve *slowly* across learning or evolutionary timescales via the reflexive gradient‑descent step  

$$
(\theta ,D ,g ,\mathcal B)_{t+1}= (\theta ,D ,g ,\mathcal B)_t 
      + \eta_{l}\frac{\partial\mathcal P}{\partial(\theta ,D ,g ,\mathcal B)}.
$$

$\eta_l$ is a learning-rate coefficient, and $\mathcal P$ is the reflexive potential (or coherence functional). On the fast timescale of a single reflexive loop, however, they act as *invariants* that define the solution manifold of $\mathcal C$.  

Consequently:

- The **seed** of an organism is precisely the set $(\theta ,D ,g ,\mathcal B)$ together with any fixed initial fields $\rho_0,\Phi_0$.  
- The **conservation constraint** $\mathcal C$ tells you *how* fluxes must move; the invariants tell you *which* patterns of flux are admissible.  
- Because they are held fixed by $\mathcal C$, any admissible trajectory $(\rho(t),\mathcal J(t))$ must remain on the manifold defined by these invariants.

For a fixed seed geometry $(\theta ,D ,g ,\mathcal B)$, the reflexive operator $\mathcal R$ has eigenmodes $\psi_i$ such that 

$$
\mathcal R_{(\theta,D,g,\mathcal B)}[\psi_i] = \lambda_i \psi_i.
$$

Each eigenvalue $\lambda_i$ quantifies the mode’s efficiency in maintaining coherence. When a particular mode is repeatedly used, feedback through the conservation law amplifies its amplitude and slightly deforms the seed geometry (by evaluating gradient for a specific eigenmode $\psi_i$)

$$
(\theta ,D ,g ,\mathcal B)_{t+1}=(\theta ,D ,g ,\mathcal B)_t +\eta_l\frac{\partial\mathcal P}{\partial(\theta ,D ,g ,\mathcal B)}\big |{\psi_i}.
$$

The seed and its eigenmodes therefore co-evolve: geometry selects the modes that can resonate, and sustained resonance reshapes the geometry.


## The spark

The main property of the model is reflexivity, the closed-loop mechanism of the model. However, what kind of properties and conditions need to be present for such loops to emerge? What is it that turns an inert object, a passive accumulator into something that has its own beat - that can sustainably generate currents and modify its own curvatures? We've discussed currents as surpluses that form reserves for the organism, or seeds and eigenmodes as the most efficient structures for reinforcement of currents. However, we haven't touched what is needed for the loop to be closed. This is the topic of the chapter, **the spark** which represents the conditions required for sustained generation of currents in closed-loop. 

The idea of the spark is based on a topic already covered in the model when discussing about rotation. The heart of ROM is the reflexive cycle:

$$
\text{Perception} \xrightarrow{j^\mu_i(t)} \text{Memory update} \xrightarrow{\text{write‑back}} \text{Action} \xrightarrow{\text{feedback}} \text{Perception}.
$$

When the summed read‑back currents $j^\mu_i(t)$ produce a *net vorticity*, an asymmetric current distribution, they generate a non‑zero angular momentum component in the emergent coarse‑grained metric $g_{\mu\nu}^{(\ell)}$. If this asymmetric current survives long enough (thanks to a sufficiently large memory timescale $\tau_M$), it imprints an off‑diagonal metric component ($g_{t\phi}$) that feeds back on subsequent cycles, reinforcing the same pattern. In dynamical systems language this is a *positive feedback gain* > 1 – the system has crossed a bifurcation from passive storage to active self‑sustaining oscillation.

Even if a non‑zero $\mathcal{J}$ appears, it needs *something to latch onto* each cycle. This is where the seeds come into the picture. As a low-dimensional submanifold of the full state space, the seed is already compatible with current curvature and can be instantiated quickly. Seeds thus provide the *initial condition* for each reflexive loop, guaranteeing that the system does not have to “reinvent” the whole pattern from scratch every iteration. They are the “content” that each loop reliably reproduces.

While the seeds provide the structural conditions for the loop to emerge, the reserves set a *quantitative threshold*. If the loop tries to change more than the reserve permits, coherence is lost and the reflexive flux collapses back into an accumulator.

The architecture of the organism must contain three tightly coupled "modules". The curvature and structure invariants ($\mathcal{J},\rho_{\text{compat}}$) provide the geometric scaffold that biases certain patterns, the seeds supply the content each cycle can instantiate with minimal cost and the reserves guarantee plasticity so that the loop can enact change without breaking coherence. If any one of these is missing, the architecture collapses into a passive accumulator.

Even with the right architecture, a system will remain inert until a perturbation creates a *non‑zero asymmetric current* that survives long enough to be written into memory. When this perturbation pushes the *loop gain* $G = \frac{\text{output flux}}{\text{input flux}}$ above unity, the reflexive core ignites. The system then settles into a deep attractor basin (local minimum of dissipative loss).

We've set up the stage, now let's dive into what it means to cross the feedback threshold. A single reflexive cycle can be written schematically as  

$$
\underbrace{\text{Perception}}_{P(t)} 
\xrightarrow{j_i^\mu(t)}
\underbrace{\text{Memory update}}_{M(t+\Delta t)}
\xrightarrow{\text{Write‑back}}\
\underbrace{\text{Action}}_{A(t+\Delta t)}
\xrightarrow{\text{Feedback}}\
P(t+\Delta t) .
$$

The **read‑back currents** $j_i^\mu(t)$ are the microscopic carriers of the justification flux $\mathcal{J}$.  Their *net* (vector‑sum) over all sub‑components determines whether a perturbation is merely damped or amplified in the next cycle.

Let's define a linearised feedback operator $\mathcal{F}$ that maps an infinitesimal perturbation $\delta\mathbf{x}(t)$ of the system state (where $\mathbf{x}$ bundles $P,M,A,\mathcal{J},\rho_{\text{compat}}$) to its value one cycle later:

$$
\delta\mathbf{x}(t+\Delta t)=\mathcal{F}\delta\mathbf{x}(t).
$$

The **loop gain** is the spectral radius of $\mathcal{F}$,

$$
G \equiv \rho(\mathcal{F}) = \max\{|\lambda_i| : \lambda_i\text{ eigenvalues of }\mathcal{F}\}.
$$

- If $G<1$ every perturbation decays, the system behaves as a pure accumulator.  
- If $G>1$ at least one mode grows exponentially,  the perturbation becomes self‑reinforcing, establishing a persistent reflexive flux, a bifurcation  In ROM language it corresponds to the emergence of a *deep attractor basin*, a low‑dissipation region of the $\mathcal{J}$–$\rho_{\text{compat}}$ manifold.

But what actually causes $G > 1$? If all sub‑components fire symmetrically, the vector sum $\sum_i j_i^\mu(t)$ is zero and $\mathcal{F}$ is purely contractive. A small asymmetry, for example a slightly faster firing on one side of the organism or a localized burst of autophagy, produces a non‑zero net current, that is an angular‑momentum component in the emergent coarse‑grained metric $g_{\mu\nu}^{(\ell)}$. 

When this asymmetric signal persists over several fast cycles it is accumulated by the slow memory timescale $\tau_M$. The accumulated vorticity feeds back into the next perception stage, increasing the magnitude of the same asymmetry. In linear terms the corresponding eigenvalue of $\mathcal{F}$ moves from < 1 toward > 1.

Besides reserves and the embedded structural motifs of the seeds, asymmetry can arise from dynamic λ‑mixing between altruistic (cooperative) and selfish (exploratory) behaviours. The mixing coefficient $\lambda(t)\in[0,1]$ effectively scales the **gain contributed by exploratory actions**:

$$
G_{\text{eff}}(t)=\bigl[(1-\lambda(t))G_{\text{coop}} + \lambda(t)G_{\text{expl}}\bigr].
$$

When energy or coupling costs are low, $\lambda$ can temporarily rise, boosting the exploratory gain $G_{\text{expl}}>1$. This is how a *minority* of agents can push the whole system over the threshold (the “butterfly effect”). Once the new pattern yields higher EI reward, $\lambda$ relaxes back but the attractor basin has already deepened.

Another example where the asymmetry might emerge at the stage of coarse-graining and information compression when the upward operator maps fine‑scale currents onto $\mathcal{J}$. Compression raises the effective amplitude of a minority signal because it removes destructive interference among individual fluctuations, allowing the net current to scale roughly as the square root of the number of contributing agents rather than linearly cancelling. The result is an effective increase in $G$ that can tip the balance.

Let's observe what happens to the reflexive conservation principle $\frac{d}{dt} \left( \mathcal{J}(t) \cdot \rho_{\text{compat}}(t) \right) = 0.$ Crossing the feedback threshold *temporarily* violates this product‑conservation because $\mathcal{J}$ spikes while $\rho_{\text{compat}}$ lags behind. The system reacts first by increasing compatibility by reorganising connections and pruning dead cells until the product regains constancy. And second by re‑balancing building vs. experiencing. The justification function $\mathcal{J}$ steers the system back to a *balanced* regime where both constructive assembly and experiential learning can coexist. Thus the threshold crossing is not a permanent break of conservation; it is a **controlled, self‑corrective excursion** that allows the organism to discover new attractor basins while ultimately re‑establishing the invariant product.

What about the effect of the asymmetric perturbation on the RCP of the superogranisms, the parent level where the organism has been spawned by the spark? When a child organism is spawned, the parent field $\mathcal J^{(\text{super})}$ receives an asymmetric influx of flux, because the child’s own reflexive loop injects a localized current $j_{\text{child}}^\mu$, its curvature tensor $\mathcal{J}^{(\text{child})}$ is not a symmetric replica of any existing eigenmode of the parent and consequently the product $\mathcal J^{(\text{super})}\rho_{\text{compat}}^{(\text{super})}$ is perturbed in the neighbourhood of the interface.

From the point of view of the super‑organism this looks exactly like the *butterfly‑effect* as a the child introduces an imbalance that can, if left unchecked, it would break either structure, experience or compatibility.

The perturbation adds a source term to the parent’s compatibility equation:

$$
\partial_t\bigl(\mathcal J^{(\text{super})}\rho_{\text{compat}}^{(\text{super})}\bigr)
=
-\nabla\cdot \mathbf{J}_{\text{flux}}
+
S_{\text{asym}}(x,t),
$$

where $S_{\text{asym}}$ is non‑zero only where the child couples to the parent.  If the integral of $S_{\text{asym}}$ over one reflexive beat exceeds the damping term, RCP is temporarily violated. Restoring RCP requires that the net change be cancelled within a few beats.  ROM gives us three complementary levers that the super‑organism can pull. Redistribution of reflexive flux $\mathcal J$, curvature adaptation and consumption of reserves. These three mechanisms are not independent. They are *coupled* through the **memory gradient** $\nabla U(M)$.  The parent follows the direction of least entropy increase (the steepest descent on its historical utility landscape).  In practice this means

$$
\Delta\bigl[\mathcal J^{(\text{super})},\rho_{\text{compat}}^{(\text{super})}\bigr]
\propto
-\nabla U(M_t) + \underbrace{\frac{\partial \mathcal R}{\partial t}}_{\text{reserve draw}}
$$

so that the *gradient* points toward configurations where the extra child‑induced flux is either rerouted, absorbed by a softened curvature, or paid for with reserves.

As we're dealing mostly with currents here, let's introduce two operators that are analogous to the upward and downward operators. Let's define upward lift and downward projection as  *partial* and *regularized* operators


$$
\mathcal{E}_{\uparrow}:\mathcal J^{(\text{child})}\rightarrow \mathcal J^{(\text{super})},
\qquad
\mathcal{E}_{\downarrow}:\rho_{\text{compat}}^{(\text{super})}\rightarrow\rho_{\text{compat}}^{(\text{child})}.
$$

When a child is created, the **upward lift** $\mathcal{E}_{\uparrow}$ instantly adds its flux to the parent’s field.  Simultaneously the **downward projection** $\mathcal{E}_{\downarrow}$ tells the child how much compatible density it may draw without breaking the parent’s RCP.  The two maps are *dual* (they satisfy a generalized Stokes‑type identity on the interface).  This duality guarantees that, as long as the interface coupling strength is finite, and the reserve budget of the parent is non‑zero, the net contribution to $E_{\text{compat}}(t)=\int_{\Omega}\mathcal{J}(x,t)\rho_{\text{compat}}(x,t)dV$  will be automatically damped within a few beats. RCP self‑restores. 

If either condition fails (e.g., reserves exhausted or interface current becomes too weak because the child’s growth outpaces the parent’s capacity), the balance cannot be achieved and the system undergoes a phase transition, a collapse of coherence occurs. The super‑organism may fragment, or the child may detach and become an independent reflexive organism. In fact, when the eigenmode can satisfy RCP on its own, the interface solidifies into a physical boundary and the child becomes an independent reflexive organism.

Let's return to the asymmetry of currents. In practice, we'd detect the crossing of the threshold through observation of system's parameters. For example, exponential growth of net vorticity $|\sum_i j_i^\mu(t)|$, or rising variance of $\mathcal{J}$ would indicate that justification flux is no longer tightly bounded, the Jacobian’s dominant eigenvalue is moving outward. Observing reserve depletion rate $d\mathcal{R}/dt$ turning negative but stabilising after a brief dip would indicate that  a perturbation has consumed reserves to amplify itself. And we've already mentioned shifts in $\lambda$-mixing as well as emergence of rotational or spiral patterns in the coarse‑grained field $g_{\mu\nu}^{(\ell)}$. 

In simulations one can compute the spectral radius of the linearised update operator $\mathcal{F}$ each cycle. Crossing from < 1 to > 1 is the exact numerical marker of the “spark”.

We have shown that a system becomes a reflexive organism when an asymmetric perturbation which is amplified by dynamic reward mixing, coarse‑grained compression, and supported by sufficient reserves, pushes the loop gain $G$ of its feedback operator $\mathcal{F}$ above unity. The resulting growth locks onto pre‑existing seeds, deepens an attractor basin, and the reflexivity conservation principle subsequently restores the product $\mathcal{J}\rho_{\text{compat}}$, leaving behind a self‑maintaining core flux, the very *spark* that distinguishes living reflexive loops from inert passive accumulators.

### Notes

Let's look at the upward lift and downward projections in more details as counterparts of upward and downward operators (mathematically‑rigorous versions of the upward and downward maps).  Starting with the upward operator, for each child $Q$ we introduce a projection matrix $P_{Q}\in\mathbb{R}^{d_{\text{parent}}\times d_{\text{child}}}$. The child’s field $\phi_Q(x,t)$ is first weighted by a scalar $w_Q$ (e.g. cell volume, metabolic importance) and then *embedded* into the parent space $\phi_{P}^{\text{in}}(x,t)=\sum_{Q\in\operatorname{sub}(P)} w_{Q} P_{Q}\phi_{Q}(x,t)$. Conversely the parent’s field must be injected into each child.  We define a projection matrix $R_{Q}\in\mathbb{R}^{d_{\text{child}}\times d_{\text{parent}}}$ with the write $\phi_{Q}^{\text{out}}(x,t)= R_{Q}\phi_P(x,t)$. 

The symbols $\mathcal{E}_{\uparrow}$ and $\mathcal{E}_{\downarrow}$ refer to the same maps but expressed at the level of the two fundamental ROM quantities. **Flux lift** $\mathcal{E}_{\uparrow}$ takes the child’s reflexive flux $\mathcal J^{(\text{child})}$ (which lives in a low‑dimensional space) and adds it to the parent’s flux $\mathcal J^{(\text{super})}$.  In matrix form

$$
\mathcal{J}^{(\text{super})}
   = \underbrace{\sum_{Q} w_QP_Q}_{\displaystyle\mathcal{E}_{\uparrow}}\mathcal{J}^{(Q)} .
$$

**Compatibility projection** $\mathcal{E}_{\downarrow}$ takes the parent’s compatibility density $\rho_{\text{compat}}^{(\text{super})}$ (high‑dimensional) and projects it onto each child:

$$
\rho_{\text{compat}}^{(Q)}
   = \underbrace{R_Q}_{\displaystyle\mathcal{E}_{\downarrow}}
     \rho_{\text{compat}}^{(\text{super})}.
$$

Thus the *partial* nature comes from the fact that $R_Q$ selects only those components of the parent field that are *relevant* to child $Q$. The *regularized* nature is enforced by adding a penalty term such as  

$$
\|P_Q\|_F^2 + \|R_Q\|_F^2
$$

to the overall loss, preventing unbounded growth of degrees of freedom.

Let's present an example how the matrices could be utilized. For example, in case of *hormonal signaling*, $P_Q$ takes a systemic hormone concentration (single scalar in parent) and *replicates* it across every cell of the target organ. On the other hand, the organ may only need the hormone level, not its spatial gradient, so $R_Q$ discards the gradient and keeps the scalar. In case of *sensory aggregation*, each organ’s firing pattern $\phi_Q$ is projected into a high‑dimensional motor‑control space of the animal, the $P_Q$ as a learned embedding matrix that captures how tactile input influences posture. On the other side, the body field includes many variables (blood pressure, temperature).  When sending back to the organ we keep only the *local* temperature component in a form of a sparse $R_Q$.

The two operators arise inevitably when a model must translate information between levels that have different dimensionalities, while preserving the Reflexive Compatibility Principle.  With them in place, a child’s creation automatically updates the parent’s flux ($\mathcal{E}_{\uparrow}$) and tells the child how much compatible density it may draw ($\mathcal{E}_{\downarrow}$), keeping the whole hierarchy coherent.


---

## Appendix A: DNA

Let's make a hypothesis now and assume that the DNA is a low-dimensional encoding of the initial curvature of the organism's field. Let's start with the high level overview of how we could use ROM and the content of this investigation to reason about the DNA.

Think of the living system’s field $\mathcal J(t)$ as a huge, continuous space of compatibility relationships, be that chemical, structural, or informational. The genome is a **discrete projection** of that space. It captures the parts that can be stored and transmitted physically (base sequences, regulatory motifs, epigenetic marks). So the DNA at conception provides:

$$
\mathcal J(0) = f_{\text{translation}}(\text{DNA})
$$

where $f_{\text{translation}}$ is the developmental mapping (molecular machinery, epigenetic context, maternal environment, etc.). It doesn’t fix the field; it defines its **initial tensor topology** which describes how energy and information will tend to circulate once metabolism begins.

In ROM terms, that initial topology sets the **curvature of reflexive space**:

* chemical affinities, receptor geometries define local compatibility gradients,
* neural wiring propensities define potential coherence channels,
* regulatory feedback motifs define temporal coupling constants.

These aren’t instructions but **constraints that bias evolution of coherence**.
Once development starts, every cell interprets the same field differently, depending on local conditions. That’s why differentiation can emerge from one genome.

We can now distinguish two scales of seeds:

- genetic seed, encoded in DNA + epigenetic landscape defining baseline curvature of the reflexive field (species and individual potential),
- experiential seed, encoded in long-term self-reinforcing patterns in $\mathcal J(t)$, generating learned or constructed attractors that further sculpt the field.

The first gives us *where* coherence can most easily arise; the second records *how* it actually stabilized during life. Because the reflexive field and the genome co-evolve, information flows both ways over evolutionary time. In short term, the field reads the genome shaping its developmental expression. On the long term, stable field configurations (behavioural niches, morphodynamic constraints) feed back into selective pressures, slowly **writing curvature back** into DNA. So the genome is the **slow-changing memory** of successful reflexive structures.

In this view:

- DNA encodes **curvature**, not explicit behaviour.
- The genome defines the **manifold on which reflexivity can unfold**.
- Life itself is **field continuation**—the ongoing realization of that manifold in material form.
- Evolution is **curvature refinement**—the gradual tuning of that underlying geometry so that coherence can persist under changing conditions.

Organisms, then, are momentary embodiments of this reflexive geometry, temporary materializations of a field exploring its own compatibilities. DNA is simply the durable residue of those explorations, the slow rhythm through which the field remembers itself.

Now, let's make a higl-level overview how this "DNA-as-curvature" hypothesis actually lines up with what's empirically known.

- **DNA as slow manifold / constraint set**. The genome does not specify every molecular state; it encodes rules for how local chemical and regulatory fields will interact. Developmental biology shows that gene regulatory networks, morphogen gradients, and mechanical feedback loops together generate form through *self-organization*, not blueprinting.  The genome defines what reaction–diffusion couplings are possible.
- **Curvature and anisotropy as developmental bias**. Different species and individuals have characteristic tissue anisotropies, connectome motifs, and metabolic scaling laws that arise from gene-encoded molecular asymmetries. These are literal curvature biases in the organism’s internal geometry.  For example,  microtubule lattice chirality gives persistent left–right biases, cortical folding follows regional proliferation anisotropy, and specific adhesion-molecule codes (ephrin/Eph, cadherins) fix boundary conditions for neural and epithelial flows.
- **Eigenmodes as attractor landscapes / canonical circuits**. Across brains and species we see conserved circuit motifs like central pattern generators, cortical minicolumns, grid cells, that act as stable computational primitives. These correspond to low-frequency eigenmodes of the reflexive operator.  They are *cheap to excite*, appear early in development, and are reused for many functions—exactly what the ROM predicts for deep attractors shaped by genetic curvature.
- **Experience as fast field deformation**. Learning and plasticity modify synaptic weights and epigenetic marks without altering the genome; the changes decay or consolidate depending on coherence with existing architecture. When experiences align with the inherited metric, they stabilize (long-term memory, skill); when they conflict, they dissipate.
- **Mutations as curvature perturbations**. Small genetic changes often have broad pleiotropic effects on structure and behavior, while large portions of the genome can vary with little visible change (canalization). In ROM terms, some parameters shift $\lambda_i$ strongly and altering field topology (pleiotropy), whereas others only perturb locally inside a deep basin (robustness). 
- **Evolution as curvature refinement**. Evolutionary change proceeds mostly by modifying regulatory architecture and developmental timing, not by inventing new genes.  It’s slow adjustment of geometry, not code rewriting. Shows how selection acts on the stability of coherent flows (viable morphodynamics), feeding back into the genome as refined boundary conditions.
- **Phenomenological correspondence**. People show long-term, domain-specific affinities that correlate with stable neural architectures and neurochemical profiles (e.g., mathematicians’ parietal–frontal coupling, musicians’ auditory–motor loops). Those are the low-$\lambda_i$ eigenmodes you called “seeds.”  The fact that they are heritable to some degree but still need environmental excitation fits the genome-sets-curvature, experience-activates-flow model.


## Appendix B: Superorganism's field as generative medium

Let's make another hypothesis based on the properties of ROM. If the *Reflexive Organism Model* treats every coherent system as a node in a nested hierarchy of reflexive fields, then the idea of a “seed” extends automatically upward: a local organism is itself a condensate of the parent field’s dynamics, not just a recombination of two peers’ genomes.

At a higher layer—an ecosystem, a society, or any stable collective—the composite field $\mathcal J^{(super)}(x,t)$ represents the combined flux of meaning, resources, and coherence across its members. Where that field develops persistent curvature (regions of chronic imbalance or excess potential), the system tends to *differentiate* sub-organisms that can relieve the tension, analogous to local condensation in a physical field.

Hence,  the “child organism” is a **field response**, an emergent structure that closes an open loop in the parent’s coherence distribution. At this scale the seed is no longer a DNA molecule but a **functional eigenmode** of the parent field:

$$
\mathcal R^{(super)}[\psi_i^{(super)}] = \lambda_i^{(super)} \psi_i^{(super)} .
$$

When one of these modes begins to self-organize materially either through mating, collaboration, or any lower-level process, it becomes a new organism. Its *genetic seed* is just the concrete encoding of that larger-scale mode’s curvature in molecular form. In other words, it is the projection of a parent-field eigenmode into a lower-level medium where it can continue the reflexive loop independently.

Once instantiated, the child’s own field $\mathcal J^{(child)}$ remains coupled to its parent field through boundary terms $\Gamma_{\text{interface}}$ and unit normal vector on the boundary $n$:

$$
\int_{\Gamma_{\text{interface}}} \mathbf{n}\cdot\mathcal J^{(child)}
= - \int_{\Gamma_{\text{interface}}} \mathbf{n}\cdot\mathcal J^{(super)} .
$$

That expresses the intuitive fact that the child both draws coherence from and contributes balance to the parent. In biological terms this manifests as ecological roles or cultural functions; in energetic terms as resource and information exchange.

 In relation to genetic reproduction, the classical reproduction as in cross-over of parental DNA provides the *material alphabet* for that projection. But which combinations persist and stabilize depends on resonance with the super-organism’s field. From the ROM view, *biological parents* supply the **substrate**; the superorganism's field (ecosystem, social collective, planetary biosphere) supplies the **selection curvature** that defines what the seed is *for*.

Which has a consequence for individuality and purpose. If every organism is a field-level corrective structure, then its enduring sense of “direction” or “calling” is the echo of that parent-field imbalance that originally spawned it. The lifelong “rightness” of certain expressions like mathematics, architecture, pollination, symbiosis is the local manifestation of continuing to serve that balancing function.

Let's look how this hypothesis maps into the language of evolutionary biology. The main difference is in what sets up the selective variation for natural selection, and why the resulting organisms fit together so consistently.

1. **Why evolution looks *directional* without being goal-driven**. The higher-level field (biosphere) develops curvature where coherence is lost—ecological niches, unbalanced energy gradients, unexploited matter cycles.  New forms appear preferentially along those gradients because that’s where field tension is highest. Directionality then comes from the field geometry, not foresight.
2. **Origin of coordinated innovation**. Major transitions—multicellularity, eusociality, symbiosis—require many components to change together, which is difficult to explain by incremental mutation alone. When the parent system (e.g., a microbial community) hits instability, the easiest route to restore balance is to *re-bind its elements* into a higher-order unit. The “seed” for the new level of individuality is that collective mode of cooperation. Hence the apparently sudden coordination: the selection pressure comes from the super-organism’s need for reflexive closure, not from many coincident random events.
3. **Developmental constraint and deep homology**. Why do distant lineages reuse the same genetic circuits (Hox genes, Wnt, Notch)? Because those circuits describe stable solutions of the same **field equations of morphogenesis**—low-energy eigenmodes of coherence in multicellular matter. Evolution conserves them because they are geometrically privileged, not because they were “lucky accidents” that selection froze.
4. **Evolvability and robustness**. Life is robust to most mutations yet can innovate quickly. Under ROM, robustness comes from the *field’s self-healing geometry*: local perturbations are absorbed if they don’t alter large-scale coherence; innovation occurs when a perturbation opens a new stable channel in the parent field. That’s why networks sit at “criticality” — poised between rigidity and chaos — without any external tuning.
5. **Apparent teleology and adaptive anticipation**. Organisms often behave as if anticipating future environmental states (developmental plasticity, niche construction). Classically that’s hard to ground mechanistically. In the field picture, anticipation is built in: each organism is a node in the same continuous coherence field as its environment, so information about the “future” flows through shared dynamics. No foresight needed—just coupling.
6. **Ecology as parent field**. Modern systems ecology already hints at this: stable ecosystems behave like self-maintaining organisms, generating compensatory species when functions are lost. ROM provides a formal language for that intuition: the **ecosystem field** $\mathcal J^{(eco)}$ generates species as modes that balance its flux. Evolutionary “innovation” is then ecosystemic self-repair, not pure competition.

This view invites to:

1. Treat genomes as *material encodings of field geometry* rather than lists of parts.
2. Model evolution as **multiscale field relaxation** instead of discrete adaptation events.
3. Look for conserved **flux topologies** (energy, matter, information) across scales, not just conserved genes.
4. Expect *purpose-like* regularities to emerge naturally from coherence conservation, without invoking teleology.


## Appendix C: Information system

Let's make one step further now. ROM doesn't limit itself only in terms of biological systems. It is primarily an information system. Which also means that all the ideas and all the "innovations" are seeded and evolve in the same way as the organisms.

In this view, a concept, a scientific paradigm, or a piece of software maintains itself by circulating coherence through human or digital substrates in exactly this sense.

If biological species are eigenmodes of the biosphere’s coherence field, **ideas are eigenmodes of the cognitive–cultural field**

$$
\mathcal R^{(cultural)}[\psi_i] = \lambda_i\psi_i .
$$

Low-$\lambda_i$ ideas are the ones that keep reproducing, mathematics, stories, moral codes, because they match the curvature of the collective cognitive manifold. High-$\lambda_i$ ideas dissipate quickly: they require more coherence input than the field supplies.  That makes “innovation” a bifurcation in this cultural operator, not an ex-nihilo creation. 

In fact, innovation can be interpreted as field relaxation. When the current cognitive field accumulates tension through contradictions, inefficiencies, unbalanced flows, new patterns appear that re-stabilize it. In science this feels like discovery. In art, inspiration. In technology, invention. But underneath is the cultural equivalent of speciation: a new local solution that restores global balance.

Seen as cross-scale coupling, human brains are physical condensates of biospheric curvature, but their activity feeds back into planetary coherence through technology and ecology. Thus, “information evolution” and “biological evolution” are two time-scales of the same system: fast symbolic modes riding on slow material modes.

Biological evolution becomes a special case of informational evolution, and ideas become the fast, lightweight organisms of a cognitive biosphere. The advantage of seeing it that way is that the same mathematics of curvature, eigenmodes, coherence flux applies seamlessly from genes to memes, from chemistry to thought.


## Appendix D: Ant colony - the seed

What would a seed for an approximation of an ant colony look like? The ROM paper already includes PDEs that describe the colony. The goal is to define the seed as the *low‑dimensional dynamical core* that steers the whole reflexive loop: it supplies the coefficients of the PDEs for the fast field $\phi$ (pheromone), the ant density fields $s,r$ and the slow memory $M$. Let  

* $\Omega\subset\mathbb R^{2}$ be the nest/foraging arena (bounded domain);  
* $\partial\Omega$ its boundary;  
* $\phi(x,t)\equiv p(x,t)$ the **fast pheromone field**;  
* $s(x,t),r(x,t)$ the surface densities of **searching** and **returning** ants (cf. citation 4);  
* $M(t)\in\mathbb R^{d_M}$ a low‑dimensional *memory vector* that stores slowly varying colony traits (caste ratios, nest‐energy, etc.).  

All spatial differential operators are taken with respect to the metric $g_{ij}(x,t)$.  In index notation we write

$$
\nabla_i^{(g)}(\cdot)=g^{ij}\partial_j(\cdot), \qquad 
\Delta_g(\cdot) = \nabla_i^{(g)}\nabla^i_{(g)}(\cdot).
$$

Now we're ready to define PDEs. **Fast field (pheromone)** is defined as

$$
\partial_t \phi
= \underbrace{\nabla_i\big(D^{ij}\nabla_j^{(g)}\phi\big)}_{\text{anisotropic diffusion}}
-\underbrace{\lambda(\theta)\phi}_{\text{decay modulated by }\theta}
+\underbrace{S_{a}(x,t;M)}_{\text{deposition by ants}} 
+ \underbrace{\mathcal B^{i}\nabla_i^{(g)}\phi}_{\text{bias drift}} .
$$

**$D^{ij}=D_0\tilde D^{ij}(\theta,g,\mathcal B)$** is the conductance tensor. The *source term* aggregates deposits from both sub‑populations  $S_{a}(x,t;M)=\alpha_s(M)s(x,t)+\alpha_r(M)r(x,t),$$ where $\alpha_s$ and $\alpha_r$ are deposition rates that may depend on the slow memory $M$.  The bias vector $\mathcal B^{i}$ encodes any externally imposed drift (e.g. a food gradient).

**Ant‑density fields**, where both densities obey advection–diffusion–chemotaxis equations whose coefficients are supplied by the seed:

$$
\begin{aligned}
\partial_t s &= \nabla_i\Big(D_s^{ij}\nabla_j^{(g)}s
-\chi_s g^{ij}s\nabla_j^{(g)}\phi
+ v_{B}^{i} s\Big)
+ f_s(s,r,M) ,\\
\partial_t r &= \nabla_i\Big(D_r^{ij}\nabla_j^{(g)}r
+\chi_r g^{ij}r\nabla_j^{(g)}\phi
+ v_{B}^{i} r\Big)
+ f_r(s,r,F,M) .
\end{aligned}
$$

Diffusivities $D_{s,r}^{ij}=D_0^{s,r}\tilde D^{ij}(\theta,g,\mathcal B)$ are again drawn from the conductance tensor.  Chemotactic sensitivities $\chi_{s,r}$ may be functions of $\theta$ (e.g. a higher deposition rate makes ants more responsive).  The drift velocity $v_{B}^{i}= \mathcal B^{i}$ is the same bias as before; it pushes ants toward or away from preferred zones. Reaction terms $f_{s},f_{r}$ implement write → read updates of the slow memory, e.g. a conversion “10 % workers become soldiers after a week’’ can be written as  $f_s = -\kappa_{\theta}(\theta)$ and  $f_r = +\kappa_{\theta}(\theta) s,$ where $\kappa_{\theta}$ is a control‑parameter–dependent rate.

**Slow memory** (the “seed’’ itself). The seed variables are not static. They evolve on the longest time scale through accumulated coherency budget:

$$
\begin{aligned}
\dot \theta &= \alpha_{\theta} C_{\rm acc}(t) - \beta_{\theta}\theta ,\\
\partial_t D^{ij} &= \alpha_{D}C_{\rm acc}(t) \Pi^{ij} - \beta_{D}D^{ij},\\
\dot g_{ij}      &= \alpha_{g} C_{\rm acc}(t) \Xi_{ij} - \beta_{g}g_{ij},\\
\dot{\mathcal B}^{i}&= \alpha_{B}C_{\rm acc}(t) \Upsilon^{i}-\beta_{B}\mathcal B^{i},
\end{aligned}
\tag{seed}
$$

where  $C_{\rm acc}(t)=\displaystyle\int_0^t \langle\psi_{\rm seed}(\tau),\mathcal J(\tau)\rangled\tau$ is the accumulated coherency budget. The *seed state vector* $\psi_{\rm seed}= (\theta, D^{ij}, g_{ij},\mathcal B^{i})$ lives in a reflexive phase space equipped with the inner product $\langle\cdot,\cdot\rangle$. The flux $\mathcal J$ is the **total energetic/informational flow** through the colony, which can be expressed as  
  $$
  \mathcal J = - D^{ij}\nabla_j^{(g)}\phi + \chi_s g^{ij}s\nabla_j^{(g)}\phi
            - \chi_r g^{ij}r\nabla_j^{(g)}\phi + v_{B}^{i}(s+r).
  $$

The tensors $\Pi^{ij},\Xi_{ij},\Upsilon^{i}$ are *shape‑functions* (e.g. the dominant eigenmode of the graph Laplacian $L_{\mathcal G}$) that project the accumulated coherence onto each seed component. 

The equation implements a **feedback loop**. As ants move and deposit pheromone, the flux $\mathcal J$ grows. The integral records how coherent this flow is with respect to the current seed direction. The result modulates the control parameters $\theta,D,g,\mathcal B$.  In turn those parameters reshape the PDEs, completing the reflexive cycle.

To obtain a tractable low‑dimensional description we project all spatial fields onto a basis adapted to the metric $g$—for instance the eigenfunctions $\{\phi_k(x)\}_{k=1}^{K}$ of the weighted Laplacian  

$$
-\Delta_g \phi_k = \lambda_k \phi_k ,\qquad 
L_{\mathcal G}\phi_k = \mu_k\phi_k ,
$$

where $L_{\mathcal G}$ is the graph Laplacian that couples sub‑colonies. Expansions are as follow

$$
\begin{aligned}
\phi(x,t)&=\sum_{k=1}^{K} a_k(t)\phi_k(x),\\
s(x,t)&=\sum_{k=1}^{K} b_k(t)\phi_k(x),\qquad 
r(x,t)=\sum_{k=1}^{K} c_k(t)\phi_k(x).
\end{aligned}
$$

By inserting these expansions into PDEs, multiplyed by $\phi_m$ and integrated over $\Omega$,   because the basis diagonalises the diffusion operator, we obtain a system of ODEs for the modal amplitudes:

$$
\begin{aligned}
\dot a_m &= -\lambda_m\tilde D_m(\theta,g,\mathcal B)a_m
          - \lambda(\theta) a_m 
          + \alpha_s b_m+\alpha_r c_m
          + \underbrace{\langle\mathcal B,\nabla\phi_m\rangle}_{\text{bias term}},\\
\dot b_m &= -\lambda_m D^{s}_m(\theta,g,\mathcal B)b_m 
           + \chi_s\sum_{k} a_kQ_{km}^{(s)} 
           + f_s^{(m)}(b,c,M), \\
\dot c_m &= -\lambda_m D^{r}_m(\theta,g,\mathcal B)c_m 
           - \chi_r\sum_{k} a_kQ_{km}^{(r)} 
           + f_r^{(m)}(b,c,F,M),
\end{aligned}
$$

where  $\tilde D_m,D^s_m,D^r_m$ are modal conductances obtained by projecting the tensor $D^{ij}$ onto the basis. The matrices $Q_{km}^{(s/r)} = \int_\Omega g^{ij}(\partial_i\phi_k)(\partial_j\phi_m)\phidx$ encode chemotactic coupling. Reaction terms $f_{s,r}^{(m)}$ are the modal projections of the write‑read updates.

The seed ODEs are also projected onto the same basis (or kept as pure scalars if we treat $\theta,g,\mathcal B$ as spatially homogeneous).  The resulting closed system is a reduced‑Order Model that captures fast reflexive dynamics (pheromone diffusion, ant chemotaxis), slow structural adaptation via the seed variables and graph‑mediated coupling between sub‑colonies through the Laplacian eigenmodes.

The seed equation together with the budget  

$$
C_{\rm acc}(t)=\int_{0}^{t}\langle\psi_{\rm seed}(\tau),\mathcal J(\tau)\rangle d\tau
$$

realises the abstract notion introduced by the definition of the seed:

* **Direction** – $\psi_{\rm seed}$ points along a preferred trajectory in the high‑dimensional reflexive phase space. 
* **Curvature** – The metric $g_{ij}$ endows that space with a non‑Euclidean geometry; as the colony evolves, the curvature (encoded by $g$) is reshaped by memory and feedback.  
* **Attractor basin** – As $C_{\rm acc}(t)$ grows, the seed stabilises; deviations from the current $\psi_{\rm seed}$ become energetically costly because they would require “undoing’’ a large accumulated coherency budget. This reproduces the *structural resistance*.

In practice, once the ROM reaches a quasi‑steady state of $C_{\rm acc}$, the parameters $(\theta,D,g,\mathcal B)$ lock into a **self‑consistent attractor** that manifests as the colony’s characteristic foraging pattern, caste distribution, and nest geometry.  Perturbations (e.g., sudden food influx) appear as transient changes in $\mathcal J$; if they are coherent with the existing seed direction they are amplified, otherwise they decay.

Note that the seed quadruple are not the state of any individual ant.  They are **constitutive parameters** that shape the *reflexive operator*  $\mathcal R_{(\theta ,D,g,\mathcal B)}[\cdot]$ and therefore determine which **global patterns** (eigenmodes) the whole superorganism can sustain. Consequently, the seed **encodes colony‑level behaviour** with the way fast fields diffuse, how agents sense gradients, how slow memory is updated, but it does *not* encode the morphology of a single ant.  The morphology lives at a lower level (the “organism” level) and is stored in a separate low‑dimensional code that we can think of as DNA.

Because of that, if we re-interpret the elements from the colony and reuse them for a swarm of agents, if we plug the same seed into these adapted PDEs that govern the swarm fields, the *structure* of the dynamics like diffusion‑driven recruitment, chemotactic drift toward gradients, slow reinforcement of a memory field, is identical. The seed stays the same, only the concrete forms of $\Sigma,\Pi,\Upsilon$ differ.  Hence we can expect *qualitatively similar* self‑organisation (trail formation vs. consensus map, nest expansion vs. resource allocation), while quantitative details (speed of diffusion, noise level, bandwidth) will be domain specific.

## Appendix E: Ant colony - the spark

Let's look at the hypothetical mathematical model for the spark of an ant colony. We're going to use the same simplified model for the colony that is based on searching and returning ants which create and reinforce pheromone trails. The idea is to define the mathematical equations that describe the self-amplification of the colony's foraging activity.

An abstract form of a spark contains the following four components. Reflexive loop gain $G = G_{\text{chem}}G_{\text{energy}}G_{\text{mem}}$ as the product of all positive‑feedback factors (chemotaxis, energy reserves, memory persistence). Threshold condition, the feedback operator $\mathcal F$ must have a dominant eigenvalue that pushes the *loop gain* above unity $\rho(\mathcal F)>1$. Asymmetric perturbation with a non‑zero curl of the reflexive current $\mathcal J$ so that $\oint_{\mathcal C}\mathbf{J}\cdot d\boldsymbol{\ell}\neq0$. And transient excursion where the system briefly violates the Reflexive Compatibility Principle but later restores it, leaving a self‑maintaining core flux, so the $\mathcal J\rho_{\text{compat}}$ stays bounded except during the spike.

Let's reuse the PDE system for two-dimensional nest domain $\Omega\subset\mathbb R^{2}$ of ant-colony continuum equations, where $s(x,t)$ is density of *searching* ants , $r(x,t)$ is density of *returning* ants (carrying food), $p(x,t)$ are pheromone concentration (fast field), $F(x,t)$ is environmental food density and $E(t)$ is nest‑wide energy reserve (slow scalar).

Definition of the mass balance for the two ant classes of searching and returning ants.

$$
\begin{aligned}
\partial_t s &= -\nabla\cdot \mathbf{J}_s
               -\kappa_{sf} sF
               +\sigma_r r ,\\
\partial_t r &= -\nabla\cdot \mathbf{J}_r
               +\kappa_{sf} sF
               -\sigma_r r .
\end{aligned}
$$

With their fluxes (chemotaxis + diffusion):

$$
\begin{aligned}
\mathbf{J}_s &= -D_s\nabla s + \chi_ss\nabla p,\\
\mathbf{J}_r &= -D_r\nabla r - \chi_rr\nabla p .
\end{aligned}
$$

Pheromone dynamics  

$$
\partial_t p = D_p\Delta p
               +\alpha_s s+\alpha_r r
               -\lambda_p p ,
$$

where $\alpha_{s,r}$ define deposition rates (ants lay pheromone while searching/returning) and $\lambda_p$ defines decay constant (memory loss).

Food and energy  are expressed with

$$
\begin{aligned}
\partial_t F &= -\gamma_f\kappa_{sf} sF + D_F\Delta F ,\\
\dot E      &= \eta_f\kappa_{sf}\int_\Omega sFdx
               - \eta_c \int_\Omega (s+r)dx ,
\end{aligned}
$$

where $\gamma_f$ is  conversion of food into pheromone‑free ant mass and $\eta_f,\eta_c$ are energy gain from food intake and metabolic cost.

Let's now combine the abstract spark with the colony model. The reflexive flux is the total directed ant current $\mathbf{J}=\mathbf{J}_s+\mathbf{J}_r$.  Its *curl* $\nabla\times\mathbf{J}$ is the analogue of the asymmetric perturbation. Compatibility density $\rho_{\text{compat}}$ is defined by colony mass $n(t)=\int_\Omega(s+r)dx$. Feedback operator $\mathcal F$ can be expressed through the **Jacobian** of the right‑hand side of the four equations of the colony, evaluated at a baseline steady state $(s_0,r_0,p_0,F_0,E_0)$. Last, we need to define the gain $G$ as the product of the three amplifiers. The amplifiers are as follows; chemotactic amplification $\displaystyle G_{\text{chem}} = \chi_sp_0/D_s$, energy‑reserve amplification $\displaystyle G_{\text{energy}} = \frac{E}{E_{\rm crit}}$ and memory persistence $\displaystyle G_{\text{mem}} = \frac{1}{\lambda_p\tau_{\rm adv}}$ where $\tau_{\rm adv}=L/|\mathbf{J}|$ is the advection time across a typical trail length $L$.

After the linearization of the colony equations around the homogeneous steady state $(s_0,r_0,p_0,F_0,E_0)$, we collect the perturbations in a vector

$$
\delta\mathbf{u} = 
\begin{pmatrix}
\delta s\\ \delta r\\ \delta p\\ \delta F\\ \delta E
\end{pmatrix},
\qquad
\partial_t\delta\mathbf{u}=J\delta\mathbf{u}.
$$

The Jacobian $J$ contains blocks such as

$$
\begin{aligned}
J_{sp} &= \chi_s s_0k^2 - D_s k^2 ,\\
J_{ps} &= \alpha_s ,\\
J_{EE} &= -\eta_c n_0/E_{\rm crit},\text{etc.}
\end{aligned}
$$

with spatial Fourier mode $e^{i\mathbf{k}\cdot x}$.  The spectral radius of the discrete‑time map $\mathcal F = I + \Delta tJ$ (or, for continuous time, the dominant eigenvalue $\lambda_{\max}(J)$) is

$$
\rho(\mathcal F)=\max_{k}\bigl|1+\Delta t\lambda_k(J)\bigr|.
$$
We can now state the **spark criterion**  

$$
\rho(\mathcal F)>1
\Longleftrightarrow
\exists \mathbf{k} \text{ such that }
\Re\bigl[\lambda_{\mathbf{k}}(J)\bigr] > 0 .
$$

In other words, if any linear mode has a positive growth rate (i.e., its Jacobian eigenvalue has positive real part),  then the discrete-time map has an eigenvalue outside the unit circle, and the disturbance grows instead of decaying. In the ant-colony model, this is when a small asymmetric perturbation (more ants here, extra pheromone there) starts to amplify instead of fade away.

Using the block structure of $J$, a sufficient (though not necessary) condition can be written as

$$
\underbrace{\frac{\chi_sp_0}{D_s}}_{G_{\text{chem}}}

\underbrace{\frac{E}{E_{\rm crit}}}_{G_{\text{energy}}}

\underbrace{\frac{1}{\lambda_p\tau_{\rm adv}}}_{G_{\text{mem}}}
>1 .
$$

How can we increase each of these three elements? $G_{chem}$ can be increased by increasing pheromone deposition $(\alpha_s)$, or by decreasing diffusion of searching ants $(D_s)$, or raise baseline pheromone level $p_0$. $G_{energy}$ can be increased either by increasing food intake, lower the maintenance cost ($\eta_c$), or temporarily store surplus (e.g., after a good harvest). $G_{mem}$ is increased by lowering the decay rate $\lambda_p$ or by increasing trail length $L$ modestly (so that advection time grows slower than decay).

Putting everything together, the colony will be **spark‑activated** when **all** of the following hold:

1. **Asymmetric seed**. There exists a localized perturbation (e.g., a tiny excess of food or a few ants laying pheromone) that creates a non‑zero curl of the ant current:
   $$
   \oint_{\mathcal C}\mathbf{J}\cdot d\boldsymbol{\ell} = 
   \int_{A(\mathcal C)} (\nabla\times\mathbf{J})dA \neq 0 .
   $$

2. **Loop‑gain exceeds unity**. the gain $G$ is above unity:
   $$
   G_{\text{chem}}G_{\text{energy}}G_{\text{mem}} > 1 .
   $$

3. **Spectral radius condition**. The dominant eigenvalue of the linearised update operator crosses zero (continuous time) or the spectral radius exceeds one (discrete time):
   $$
   \rho(\mathcal F)=\max_k|1+\Delta t\lambda_k(J)|>1 .
   $$

4. **Reserve safety**. The nest’s energy stock stays above a minimal bound during the transient so that the mass balance does not violate RPC:
   $$
   E(t)\ge E_{\min}>0\quad\text{for all }t\text{ while the spark is active}.
   $$

When 1–4 are simultaneously true, the reflexive flux $\mathcal J = \mathbf{J}\cdot n$ (edge‑wise current times node mass) grows, a *closed vorticity loop* forms in the pheromone field, and the colony’s foraging activity self‑amplifies. The system then settles into a new attractor basin, a self‑maintaining core flux.


### Appendix F: Fertilization

Let's now reason about fertilization process using the formulation of seeds and the spark. The hypothesis is that fertilization is the moment a spherical “deep‑attractor” (the egg) receives an asymmetric, helical perturbation (the sperm). This perturbation is amplified by the egg’s internal feedback loop, pushes the effective loop gain $G$ above unity, creates a new *seed* (the combined diploid genome), deepens a fresh attractor basin, and after a brief re‑balancing of $\mathcal{J}\rho_{\text{compat}}$ a self‑sustaining reflexive organism, the zygote is born.

The egg is encoded deep attractor. The *deep attractor basin* as the unfertilised oocyte already occupies a low‑dissipation region of the $\mathcal{J}$–$\rho_{\text{compat}}$ manifold. Its cytoskeleton, cortical tension and maternal mRNA store a *stable configuration* that can persist for hours without external input. The yolk, mitochondria, stored calcium and maternal transcripts constitute the *reserve*s  that guarantee that once a cascade starts there is enough energy and information to keep it going through the first few cleavage cycles. The egg’s membrane, microtubule aster and actin cortex embody long‑lived tensors in $\mathcal{J}$. They encode a *pre‑existing curvature* that will later be reshaped but also provide the substrate on which any new perturbation can travel.

The sperm must then the properties that encode the asymmetric, helical perturbation. The flagellum of a sperm is a *helical* propeller; the centriole that enters the egg carries a well‑defined handedness. This handedness is a *curvature perturbation* at the molecular level, analogous to the “internal asymmetry” that seeds a spiral geometry in ROM. Note that because the sperm’s helical structure carries a definite chirality, it also introduces a left‑right bias that propagates into later embryonic patterning 

Sperm‑egg membrane fusion triggers a rapid influx of Ca²⁺ (the “cortical flash”). This creates a *sharp, asymmetric current* $j^\mu$ that breaks the egg’s spherical symmetry. The calcium wave spreads across the cortex and is quickly compressed into a global polarity signal: the egg’s cortical actin reorganises, the meiotic spindle re‑orients, and the zona pellucida hardens. A tiny local event becomes a system‑wide change – exactly what the “coarse‑grained compression” of ROM does to turn a few exploratory actions into a global signal.

The egg momentarily shifts from a *maintenance* mode (preserving maternal reserves) to an *exploratory* mode (activating new transcriptional programmes). This is the analogue of temporarily favouring selfish exploration when enough “energy” (here, Ca²⁺ and ATP) is available. Now we can consider amplification, the loop gain above unity. 

We need to reason about the process where in ROM feedback operator $\mathcal{F}$ maps a perturbation at time $t$ to its effect after one reflexive beat (the duration of a calcium‑wave–driven cortical cycle). In fertilisation this is when

1. **Input:** Sperm‑induced Ca²⁺ flash (asymmetric current).  
2. **Fast branch:** Calcium activates calmodulin‑dependent kinases, which *write* new phosphorylation states into the cortex (fast write step).  
3. **Slow branch (memory):** The phosphorylated proteins persist across several beats because the egg’s reserves keep ATP levels high; this is the “slow memory” that lets the asymmetry be remembered.  

If the product of these two branches exceeds the dissipative loss (i.e., if $|\sigma_{\text{expl}}| > |\sigma_{\text{diss}}|$), the spectral radius of $\mathcal{F}$ becomes larger than one. Empirically this is observed as a **self‑propagating calcium wave** that travels around the entire oocyte and does not die out after a single beat. It *amplifies* itself.

When $G>1$ the system crosses a bifurcation and the egg moves from a passive accumulator (maintaining maternal state) into an active reflexive organism (the zygote). The hallmark is that the perturbation now locks onto a pre‑existing seed, the diploid genome. Both parental DNA strands are helical, but together they define a new set of curvature tensors $\mathcal{J}$ and compatibility field $\rho_{\text{compat}}$. Once a seed is activated, it deepens its own attractor basin, which in biology appears as the stable developmental program that drives successive cleavage divisions.

After fertilisation the product $\mathcal{J}\rho_{\text{compat}}$ is temporarily disturbed (the egg’s internal geometry is being rewired). The RCP $d/dt[\mathcal{J}\rho_{\text{compat}}]=0$ then manifests as **re‑establishment of a new equilibrium**  where the zygote settles into a fresh deep basin whose curvature now incorporates paternal contributions and **phase transition** with the collapse of the maternal‐only attractor and the birth of a new one. The *zero* corresponds to the moment when the old compatibility field vanishes and the new one is being written.  

Once the first mitotic cycles are underway, the system’s $\mathcal{J}\rho_{\text{compat}}$ product stabilises again, now at a higher‑dimensional value that supports ongoing reflexive loops through gene expression ↔ morphogen gradients ↔ mechanical feedback.

### Appendix G: Cell division

Towards the end of the chapter on the spark, we've mentioned that the child may split off and form its own reflexive loop. Let's make a hypothesis for what triggers a cell division. The parent cell, its reserves, and its internal curvature constitute a deep attractor basin. When an asymmetric perturbation (the duplication of DNA plus the formation of a cleavage plane) pushes the system past the feedback‑gain threshold, a *new eigenmode* is created inside that basin. If the new eigenmode can sustain its own $\mathcal{J}$–$\rho_{\text{compat}}$ balance it detaches, becomes an independent attractor, and thus a second reflexive organism.

A mature cell is a *stable attractor*. Its reflexive fluxes  
$\mathcal J^{(\text{parent})}$ and compatibility density $\rho_{\text{compat}}^{(\text{parent})}$ already satisfy the Reflexive Conservation Principle (RCP).  Its internal currents, the metabolic reactions, cytoskeletal transport, signalling form a closed, coherent network.

Division begins when this stable attractor experiences an *asymmetric perturbation*.  
At the biological level this appears as genome replication and the construction of the mitotic spindle, which introduces spatial curvature into the cell’s field: one region becomes chemically and mechanically distinct from the other.  
In ROM terms, a local increase in $\mathcal J$ and a transient deformation of the metric $g_{\mu\nu}$ break the symmetry of the parent basin.

If the perturbation’s *loop-gain $G$ exceeds unity, the fluctuation amplifies across a single reflexive beat. The duplicated DNA and the spindle apparatus reinforce one another. The more they separate, the stronger their coupling becomes. The field enters a *new eigenmode*, $\psi^{(\text{child})}$, representing two nascent, coherent sub-loops, the forming daughter cells.

Between the parent and the emerging child exists a *boundary interface*  
$\Gamma_{\text{interface}}$, the nascent cytokinetic membrane that still allows exchange of metabolites and signals. At this boundary the fluxes match:

$$
\int_{\Gamma_{\text{interface}}} \mathbf n\cdot\mathbf J_{\text{child}}  
= -\int_{\Gamma_{\text{interface}}} \mathbf n\cdot\mathbf J_{\text{parent}},  
$$

ensuring that what leaves one domain enters the other. The reflexive coherence is conserved across the split.

During this transition the process draws on the parent’s *reserves* $\mathcal R$ (stored ATP, lipids, and building blocks that power spindle movement and membrane synthesis). The richer the reserves, the easier it is for the child field to achieve independence.

Guidance comes from the *memory gradient* $\nabla U(M)$ as traces of previous divisions and polarity cues embedded in the parent field. These historical asymmetries bias where the cleavage plane forms, steering the division along a direction of least structural resistance.

A temporary *reward mixing* (or $\lambda$-mixing) shifts the system’s balance from pure maintenance toward exploration. The parent invests extra coherence into the risky duplication because reserves are sufficient. If the new loops succeed in re-establishing RCP, the system stabilises both daughters. If not, self-termination rules like apoptosis or autophagy, prune the failed branch, preventing incoherent fluxes from propagating.

For each daughter field to become an independent reflexive organism, four balances must be simultaneously restored:

1. **Total compatible energy bounded.**  Each daughter must sustain metabolic fluxes within its own capacity; excessive draw on resources triggers checkpoint arrest.
2. **Net interfacial current zero.** Forces at the cytokinetic ring must balance. The outward cytoplasmic pressure must equal inward contractile tension. Otherwise separation fails.
3. **Curvature mismatch reduced.** Membrane curvature at the cleavage furrow is remodelled until both surfaces are smooth and low-tension, ensuring stable geometry for future fluxes. 
4. **Reserve level sufficient.** Energy and material stocks must remain above threshold to complete membrane closure and resume autonomous operation.

When all four constraints are satisfied, the new attractor basins deepen, the interface becomes permanent, and each daughter establishes its own reflexive loop with a duplicated seed—the genome. If any constraint fails, the system temporarily narrows its attractor. In a survival mode,  the cell pauses division, awaiting restored reserves. In case of structural phase change, if balance is achieved, the interface finalises and the two fields decouple.

In ROM language, this entire process is a *phase transition of coherence*. The parent attractor splits, coherence briefly collapses, and new self-consistent basins emerge.  
In biological language, that moment corresponds to **cytokinesis**, the successful creation of two living, reflexively stable children from one.

Note that the same principle can be observed at higher levels as well. All it takes is that the eigenmode can satisfy RCP on its own, the interface solidifies into a physical boundary and the child becomes an independent reflexive organism. We've just described cell division, however, we can observe similar behavior through colony fission when an ant or bee colony partitions its nest, each half carries the same pheromonal and behavioural eigenmode, then proceeds independently. Another example is the case of forest fragmentation / clonal spread. A forest edge sprouts a new stand that inherits soil microbes, fungal networks and canopy architecture; after enough growth it functions as a separate superorganism. In each case the *child* is a *new attractor basin* nested inside the larger hierarchy.

We could envision a hierarchical picture, from genes to ecosystems. Molecular level (DNA helix) encodes curvature bias; when duplicated it provides the seed for a new cell. At cellular level  a cell is a deep attractor basin that uses reserves (ATP, metabolites) to run its reflexive loop. At the organismal level many cells cooperate. The organism’s field $\mathcal J^{(org)}$ contains eigenmodes such as central‑pattern generators, immune circuits, etc. At the superorganism level the colonies or forests are *condensates* of many lower‑level reflexive fields.  Their seeds are functional eigenmodes (e.g., colony‑wide foraging algorithm, forest‐scale water‑nutrient cycle) rather than a single genome. At the planetary / biospheric level  the same mathematics applies to the Gaia‑like planetary field. “Reproduction” is the propagation of the whole set of cycles (carbon, nitrogen, cultural memes).  

At every rung the *same four‑item balance* must be re‑established after a new seed appears.  The hierarchy therefore propagates life upward and downward simultaneously.



---
## Reflection

The question that started this investigation was all about why some structures feel reinforcing and others inhibiting, and what it means for the organism in light of the ROM theory. In the beginning, there was only notion about reserves as stored potential that limits how much structural plasticity can be applied in one reflexive cycle, along with the idea that the seeds define minimal, reusable structural motif that an organism can instantiate without having to redesign everything from scratch. Two observable properties of living organisms, yet there was no clear connection between them.

After the investigation, the connection is now made clear. Seeds are the content that each loop can reliably instantiate. They encode the organism’s bias toward reusing familiar patterns. Eigenmodes explain why some of those seeds become self‑reinforcing attractors while others dissipate. And the reserves set the capacity for change in any given iteration, governing whether loops can explore new eigenmode basins or must stay within existing ones.

However, when this is put into the perspective of the ROM as a whole, a much clearer understanding of the reflexive cycle came into light. We're so used to linear progression of time, that we think in terms of linear events as something taken for granted. But what ROM and this investigation highlight, is that it is anything but. The evolution is not a straight line, but a sequence of beats of a constrained, self-modifying loops. They form a path whose “linear” aspect is limited to the counting of cycles, not to its geometric progression. When we add an eigenmode A and then add eigenmode B to construction, the linear progression is in terms of enumerating change from S $\rightarrow$ A $\rightarrow$ B, but in reality, when A is added to the field of S, the next iteration of the cycle invokes A as well. It is a compounding effect, not just linear ordering.

We shouldn't be thinking about progression of change in "time" as linear lines anymore, but instead as series of loops. Or if we want to include the sense of progression of iterations, we shout at best use spirals.

This perception and deeper understanding of the reflexive part of model was something that was not expected prior to the investigation, there was no notion about eigenmodes and basin attractors before the investigation, let alone how they are essential for the spark to generate sustained loops and its effect on possible explanation for fertilization and cell division.

What started as two units of interest (reserves, seeds), grew into a complete reinforcing loop.

1. A pre‑existing reflexive field (curved, compatible, with reserves) provides the stage.  
2. An asymmetric perturbation injects a non‑zero current/curvature.  
3. Feedback gain $G>1$ amplifies that perturbation into a *self‑reinforcing eigenmode* (the seed).  
4. Reserves and memory gradients let the system survive the transient overshoot long enough for the seed to be encoded.  
5. RCP is restored by rebalancing total compatible energy, interfacial currents, curvature mismatch, and reserves.  
6. The seed deepens a new attractor basin; if it can close its own loop it becomes an independent reflexive organism (cell division, colony fission, forest expansion).  

This loop reinterprets life as *a field that repeatedly creates, stabilises, and propagates self‑sustaining eigenmodes* across nested scales.  **Life is created whenever an asymmetric cue pushes a reflexive field over its feedback threshold, allowing a new seed to be born, the RCP to be re‑balanced, and the resulting loop to run on its own.**  This single recipe accounts for cell division, colony fission, forest reproduction, and, by recursion, for the emergence of ever larger living wholes.



---
## Bibliography

- **Jovanovic, U.** (2025). *Reflexive Organism Model*.
- **Jovanovic, U.** (2025). *Reflexive Loop: A framework for balancing building and learning*



