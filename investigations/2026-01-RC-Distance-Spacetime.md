# Distance and spacetime in Reflexive Coherence

Copyright © 2026 Uroš Jovanovič, CC BY-SA 4.0.

## 1 Introduction: why distance needs a definition

In Reflexive‑Coherence (RC) there is no pre‑existing Euclidean background; geometry must **emerge** from the coherence field itself. The paper’s goal is to show how a meaningful notion of distance, proper time and even “spacetime growth’’ follows automatically from that field. We will treat the pair $(\Omega_{t},g_{\mu\nu}(t))$ as a smooth, positive‑definite Riemannian manifold; this provides the geometric language for distance, curvature and the notion of spacetime growth. In the end, we will also expand the continuous notion of distance to discrete version using graph theory and connect it with GRC.

## 2 Foundations of Reflexive Coherence  

In the Reflexive‑Coherence (RC) framework **all structure** is ultimately a functional of one primitive quantity, a non‑negative scalar field that we call the *coherence density*.  The sections below introduce this field, the emergent space it creates, the identity basins that partition that space, and the Riemannian geometry that makes “distance’’ and “time’’ mathematically well defined.

### 2.1 The primitive scalar – coherence density  

The only dynamical variable of RC is a smooth map  

$$
C:\Omega\times[0,T]\longrightarrow \mathbb{R}_{\ge0},
\qquad 
(x,t)\mapsto C(x,t),                                   \qquad\text{(2.1)}
$$

with the positivity constraint $C(x,t)\ge 0$.  

Every higher‑order object (organ, memory state, graph edge weight, …) is a functional of this field; no additional “state variables’’ are introduced in the coherence‑only formulation.

### 2.2 Emergent spatial support  

Because RC contains **no pre‑existing background manifold**, the region where anything can exist is defined as the *support* of the coherence field:

$$
\Omega_t :=\text{supp}\ C(\cdot,t)
          =\overline{\{x\in\mathbb R^{d}\mid C(x,t)>0\}}.
\qquad\text{(2.2)}
$$

The ambient Euclidean box $[0,1]^{d}$ (or any convenient chart) is merely a reference frame; the *actual* space of the organism at time $t$ is the set $\Omega_t$.  The topology of $\Omega_t$ is inherited from the subspace topology of $\mathbb R^{d}$.

### 2.3 Identity basins and the coherence tensor  

#### Basins as connected components  

Connected components of $\Omega_t$ are called **identity basins**:

$$
\Omega_t=\bigcup_{i=1}^{N(t)} B_i(t),\qquad 
B_i(t)\cap B_j(t)=\varnothing (i\neq j).                     \qquad\text{(2.3)}
$$

Each basin $B_i(t)$ carries its own internal geometry and a *spectral hierarchy* (see chapter 2.5). The basins are the “identities’’ of the organism. They may split, merge or disappear as the coherence field evolves.

#### Coherence tensor  

From the scalar field we build a symmetric rank‑2 tensor that contains three physically transparent contributions:

$$
K_{\mu\nu}[C] := \underbrace{\lambda_{C} C g^{(\text{aux})}_{\mu\nu}}_{\displaystyle\text{density term}} +
\underbrace{\xi_{C} \nabla_\mu C \nabla_\nu C}_{\displaystyle\text{gradient term}} +
\underbrace{\zeta_{C} j_\mu j_\nu}_{\displaystyle\text{flux term}},
\qquad
j_\mu:=C v_{C,\mu}.                                           \qquad\text{(2.4)}
$$

- *Density term* – the scalar field itself sets an overall scale.  
- *Gradient term* – spatial variations of coherence act like a “stiffness’’ that penalises sharp changes.  
- *Flux term* – coherent current $j_\mu$ (coherence times its velocity) biases geometry along active directions.

The tensor (2.4) is the bridge from the *scalar* world to the *geometric* one.

### 2.4 Induced Riemannian metric  

Dividing the coherence tensor by the local density yields a **metric**:

$$
g_{\mu\nu}(x,t) := \frac{1}{\lambda_{C} C(x,t)} K_{\mu\nu}[C](x,t). \qquad\text{(2.5)}
$$

A direct calculation shows that for any non‑zero tangent vector $u^\mu$,

$$
\begin{aligned}
g_{\mu\nu}u^{\mu}u^{\nu}
&= u^{\mu}u_{\mu}^{(\text{aux})} +\frac{\xi_{C}}{\lambda_{C}C}\left(\nabla_\mu C\,u^{\mu}\right)^{2} +\frac{\zeta_{C}}{\lambda_{C}C}\left(j_{\mu}u^{\mu}\right)^{2} \\
&>0 .
\end{aligned}
\qquad\text{(2.6)}
$$

Hence **$g_{\mu\nu}$ is positive‑definite**. The metric satisfies all axioms of a Riemannian inner product.  Consequently each pair  

$$
M_t := \left(\Omega_t, g_{\mu\nu}(\cdot,t)\right)
$$

is a **smooth Riemannian manifold**.  Moreover the map  

$$
t \longmapsto g_{\mu\nu}(\cdot ,t) 
\qquad\text{(2.7)}
$$

is smooth in the Banach space $C^{\infty}(\Omega_t)$, giving a **smoothly evolving family of Riemannian manifolds** $\{M_t\}_{t\in[0,T]}$.

Now we can reason about the following concepts.

* **Distance:** The line element $ds^{2}=g_{\mu\nu}dx^{\mu}dx^{\nu}$ supplies a local length. Integrating along any curve $\gamma$ yields its length  
  
$$
L[\gamma]=\int_{\gamma} \sqrt{g_{\mu\nu}\dot x^{\mu}\dot x^{\nu}} ds .
$$  
  The *geodesic* (the curve that extremises $L$) defines the **distance** between two points.  
* **Well‑posed PDEs:** Parabolic (gradient‑flow) equations governing the reflexive loop require a smooth, positive‑definite metric to obtain energy estimates and maximum principles.  
* **Spacetime growth as refinement:** The ambient set $\Omega_t$ never expands; “growth’’ appears only through the continuous deformation of $g_{\mu\nu}$.  Curvature wells deepen, basins split, and the induced distance network becomes richer, making it a *refinement* rather than a topological surgery.

### 2.5 Coarse‑graining via the Laplace–Beltrami spectrum  

Because every basin carries its own metric $g^{(i)}$, we may form the **Laplace–Beltrami operator** $-\Delta_{g^{(i)}}$ and expand the coherence field in its eigenfunctions:

$$
C(x,t)=\sum_{k=0}^{\infty}c_k(t)\,\phi_k(x),
\qquad 
-\Delta_{g^{(i)}}\phi_k=\lambda_k\,\phi_k .
\qquad\text{(2.8)}
$$

Low‑frequency modes ($\lambda_k$ small) describe *coarse‑grained* structure (slow, large‑scale patterns), while high‑frequency modes capture fine details and rapid reactions.  This spectral decomposition provides a natural hierarchy of internal clocks. Each basin’s slowest eigenmode defines its identity. Successive modes resolve sub‑structure, and the fastest modes encode momentary microstates.

## 3 From Coherence to Geometry  

Let's now show how the scalar coherence field $C(x,t)$ actually *creates* that metric.  This section follows the logical chain  

$$
C \;\longrightarrow\; K_{\mu\nu}[C] \;\longrightarrow\; g_{\mu\nu}[K],
$$

which is the backbone of RC.

### 3.1 The coherence tensor $K_{\mu\nu}$  

The *coherence tensor* (2.4) gathers three physically distinct contributions of the scalar field:

* **Density term** $\lambda_C C g_{\mu\nu}$: a *mass‑like* weighting – regions of high coherence “inflate’’ the metric proportionally to their density.  
* **Gradient term** $\xi_C \nabla_\mu C \nabla_\nu C$: a *gradient pressure* – sharp spatial changes act like tension that bends space.  
* **Flux term** $\zeta_C j_{\mu}j_{\nu}$: an *energy‑flux stress* – the coherent current itself pulls the geometry along its direction, turning strong currents into shortcuts.

Together these three pieces encode **how much coherence is present, how rapidly it varies, and where it flows**.  Because each term is a symmetric rank‑2 tensor, their sum $K_{\mu\nu}$ is also symmetric and can be used as the seed for a metric.

### 3.2 Metric construction  

We now **extract the metric** by a functional operation that removes any residual background dependence:

$$
g_{\mu\nu}=g_{\mu\nu}[K].
\qquad\text{(3.2)}
$$

There is **no independent, pre‑existing background metric**. The only geometry present in RC is the one induced by (2.4).  In practice the functional $g[\cdot]$ consists of a normalisation that guarantees positive definiteness and an inversion of the auxiliary Euclidean metric used only as a bookkeeping device.

The chain  $C \xrightarrow{K[C]} K_{\mu\nu} \xrightarrow{g[K]} g_{\mu\nu}$ is therefore the **complete geometric pipeline** of RC.  Because every ingredient on the right‑hand side is a smooth functional of $C$, the resulting metric inherits the same regularity:  

* $g_{\mu\nu}(x,t)$ varies smoothly in space and time;  
* the map $t\mapsto g_{\mu\nu}(\cdot,t)$ is a smooth curve in the Banach space $C^{\infty}(\Omega_t)$.  

Consequently $(\Omega_t,g(t))$ remains a **Riemannian manifold for all $t$**, ready to support geodesics and curvature.

### 3.3 Distance as geodesic length  

With a metric in hand the *line element* on $\Omega_t$ reads

$$
ds^{2}=g_{\mu\nu}(x,t) dx^{\mu}dx^{\nu}.                                 \qquad\text{(3.3)}
$$

Given two points (or, more generally, two identity basins) $x_{1},x_{2}\in\Omega_t$, the **distance** is defined as the length of the *geodesic* that extremises the functional

$$
L[\gamma]=\int_{\gamma}\sqrt{g_{\mu\nu}(x) \dot x^{\mu}\dot x^{\nu}} ds
\qquad\text{(3.4)}
$$

where $\gamma:[0,S]\to\Omega_t$ is a smooth curve with boundary conditions  $\gamma(0)=x_{1}, \gamma(S)=x_{2}$.  The extremising curve satisfies the **geodesic equation**

$$
\nabla_{s}\dot x^{\mu}=0,
\qquad\text{(3.5)}
$$

which reduces to straight‑line motion $\ddot x^{\mu}=0$ in a flat metric.  Because the metric itself depends on $C$ and its flux, the geodesic is **the least‑energy path** that respects the instantaneous flow of coherence energy.  

**Practical computation (for reference):**

1. **Snapshot** the coherence field at time $t$ → build $K_{\mu\nu}$ via (2.6).  
2. **Construct** the metric $g_{\mu\nu}=g_{\mu\nu}[K]$ using (3.2).  
3. **Solve** the geodesic equation (3.5) with the chosen endpoints.  
4. **Integrate** (3.4) along the solution to obtain the distance $d(x_{1},x_{2})$.

Repeating this procedure at a later instant yields a *different* value because the metric has been reshaped by the internal coherence redistribution, what we call **dynamic distance** or “spacetime growth’’.  In RC, growth is therefore not an expansion of coordinates but a continuous refinement of the geometry that governs how far apart any two basins are.

## 4 Local Proper Time and Hierarchical Structure  

We saw that a single scalar field $C(x,t)$ generates a smooth metric  
$g_{\mu\nu}=g_{\mu\nu}[K[C]]$ on the emergent support $\Omega_t$.  The metric is positive‑definite and therefore endows each *identity basin* $B_i(t)\subset\Omega_t$ with its own induced geometry. Now, let's make the consequences of that induced geometry explicit: every basin carries a local proper‑time clock, and distances measured across basins acquire a natural multi‑scale character.

### 4.1 Basin‑wise metrics  

For any basin $B_i(t)$ we define the **restricted metric**

$$
g^{(i)}_{\mu\nu}(x,t) := g_{\mu\nu}(x,t)\Big|_{\,x\in B_i(t)} .
\qquad\text{(4.1)}
$$

Because $B_i(t)$ is an open submanifold of the Riemannian manifold $(\Omega_t,g)$, the restriction inherits smoothness and positive‑definiteness from the parent metric.  In practice one may coarse‑average over the fast eigenmodes of the Laplace–Beltrami operator inside a basin and the resulting *effective* metric still satisfies (4.1).

### 4.2 Local proper time  

With the basin‑wise metric at hand we can introduce a **proper‑time element** for an observer that lives entirely within $B_i$:

$$
d\tau_i = \sqrt{- g^{(i)}_{\mu\nu} dx^\mu dx^\nu} .
\qquad\text{(4.2)}
$$

*The minus sign reflects the fact that each basin carries its own causal (Lorentz‑like) direction. A convention that reproduces the usual relativistic proper time when the metric is written in mixed signature.*  

A more detailed derivation shows that the factor $\frac{1}{c}$ often written in front of the square root is absorbed into the definition of the coherence‐induced metric:

$$
d\tau_i=\frac{1}{c} \sqrt{- g^{(i)}_{\mu\nu}[K[C]] dx^\mu dx^\nu}.
$$

#### Interpretation: a hierarchy of clocks  

Each basin possesses its own **spectral decomposition**  $C|_{B_i}(x,t)=\sum_k c_{i,k}(t)\,\phi_{i,k}(x)$.  
The *slowest non‑trivial* eigenmode $\phi_{i,1}$ sets a characteristic relaxation time $\tau^{(s)}_i$. This time scale is precisely what the proper‑time element (4.2) measures.  Consequently:

* Basins with **slow** spectral modes (large $\tau^{(s)}_i$) experience *more* proper‑time per unit of global coordinate time. They are “deep” in the coherence landscape.  
* Basins dominated by **fast** modes accrue *less* proper‑time. Their internal geometry is comparatively flat.

Thus **proper time becomes a local, basin‑dependent quantity**, rather than a universal background parameter.

### 4.3 Multi‑scale distance  

Because every basin carries its own clock, the notion of “distance” between two points $x\in B_i$ and $y\in B_j$ is **scale dependent**:

1. **Intra‑basin distance**: both endpoints lie in the same basin $B_i$.  The geodesic length is computed with the *local* metric $g^{(i)}_{\mu\nu}$ and measured using the proper time $\tau_i$.  
2. **Inter‑basin distance**: endpoints belong to different basins ($i\neq j$).  One must either:
   - Choose a common reference (e.g., the global coordinate time $t$) and evaluate each geodesic segment with its own metric, then concatenate the pieces; or
   - Map one proper‑time scale onto the other using the **spectral‐mode hierarchy** (the slow modes of each basin provide conversion factors).  

This leads to a **hierarchical length structure**. Short “local” distances inside high‑coherence basins coexist with longer “global” separations across low‑coherence regions.  This means that *inter‑basin distance may be measured in different clocks*.

#### Connection to dynamic refinement  

Recall from chapter 3 that the metric evolves as $g_{\mu\nu}=g_{\mu\nu}[K[C(\cdot,t)]]$.  When a basin **splits** (a new critical point of $C$ appears), a fresh proper‑time clock $\tau_{new}$ is created. When two basins **merge**, their clocks are synchronised into a single $\tau$.  In this way the *refinement* of spacetime (more basins, deeper curvature wells) automatically generates a richer hierarchy of proper times and distances.

The upshot is that **proper time is no longer a global parameter but an emergent, basin‑specific clock**.  This hierarchical timing structure underlies the multi‑scale distance network that RC predicts for living or self‑organising systems.

## 5 Temporal Evolution: Refinement vs. Expansion  

We now turn to the *time dependence* of that geometry.  In Reflexive‑Coherence (RC) “spacetime growth’’ is not an increase of the coordinate domain but a **continuous refinement** of the metric and of the basin partition.

### 5.1 Time‑dependent metric  

The metric is a functional of the *instantaneous* coherence field:

$$
g_{\mu\nu}(x,t) = g_{\mu\nu} \left[ K[C(\cdot ,t)] \right] .
\qquad\text{(5.1)}
$$

- The **coherence tensor** $K_{\mu\nu}[C]$ (eq. (2.6)) contains the density, gradient and flux contributions of the current field configuration.  
- Because $C$ evolves under the reflexive loop $C\!\to\!K\!\to\!g\!\to\!J\!\to\!C$, the metric inherits that evolution automatically.

In other words, **the geometry lives inside the coherence loop** – there is no background metric that stays fixed while $C$ changes.

### 5.2 “Growth’’ as *refinement* of basins  

In RC the ambient chart $[0,1]^{d}$ (or any other Euclidean box) never expands; only the **support** $\Omega_{t}=\mathrm{supp}C(\cdot,t)$ and its internal partition into **identity basins** evolve.

- **Split**: A connected component of $\{C>0\}$ develops a new critical point (Morse bifurcation). The original basin $B_{i}(t)$ is replaced by two children $B_{i}^{(1)}(t), B_{i}^{(2)}(t)$. The same coordinate region now hosts **more basins**. A finer partition of identity.
- **Prunning**: Two adjacent components coalesce or one disappears when its coherence falls below the viability threshold. The partition becomes **coarser** locally as total coherence is conserved.

Because the number of basins changes while the underlying point set stays fixed, we speak of **spacetime “growth’’ as refinement** rather than literal expansion.

### 5.3 Consequences for distances  

When the basin partition refines, the **metric reshapes** locally:

* **Inside a newly created or densified basin** the coherence density typically rises, deepening curvature wells.  The geodesic length between any two points *inside* that basin therefore **shrinks**, because the line element $ds^{2}=g_{\mu\nu}dx^{\mu}dx^{\nu}$ becomes smaller.  
* **In regions that lose coherence** the density term $\lambda_{C}C\,g_{\mu\nu}$ weakens, curvature flattens, and geodesic segments *expand*. The same pair of points can become nearer or farther as $g$ evolves.  
* **Directional effects**. When a strong coherent current runs through a basin, the flux term $\zeta_{C}j_{\mu}j_{\nu}$ shortens distances *along* the flow while leaving transverse directions relatively long.  Thus refinement is not isotropic. It inherits the anisotropy of the underlying energy‑flow pattern.

Putting these pieces together we obtain a **hierarchical distance landscape**:

1. **Fine‑scale basins** (high $C$, strong flux) generate short geodesics, rapid information transfer.  
2. **Coarse‑scale regions** (low $C$, weak flux) generate long geodesics, slower interaction.

Because the metric is continuously updated by (5.1), the *distance network* of the organism is a living object that contracts and expands locally while the global coordinate box remains unchanged.

## 6 Anisotropic Distance from Fluxes  

So far we've shown that the **metric** of Reflexive‑Coherence (RC) is generated entirely by the coherence tensor  

$$
K_{\mu\nu}= \lambda_C\,C\,g_{\mu\nu}
           + \xi_C\,\nabla_\mu C\,\nabla_\nu C
           + \zeta_C\,j_\mu j_\nu , \qquad 
   j_\mu = C\,v_{C,\mu},
\qquad\text{(6.1)}
$$

and that the induced metric is a smooth functional $g_{\mu\nu}=g_{\mu\nu}[K]$.  The third term in (6.1) is the only one that carries an explicit *direction*. It couples the metric to the coherent current $j^\mu$.  This section makes the geometric consequences of that coupling explicit.

### 6.1 Directional term $\boldsymbol{\zeta_C\,j_\mu j_\nu}$

#### How the term enters the line element  

Starting from the positivity proof in chapter 2.4 we can write the metric as

$$
g_{\mu\nu} = g^{(0)}_{\mu\nu} + \frac{\zeta_C}{\lambda_C C} j_\mu j_\nu ,
\qquad\text{(6.2)}
$$

where $g^{(0)}_{\mu\nu}$ collects the density and gradient contributions ($\lambda_C C g_{\mu\nu}+\xi_C\nabla_\mu C\nabla_\nu C$).  Equation (6.2) shows that **any displacement with a component along the current acquires an extra quadratic contribution** proportional to $|j|^2$.

If we choose a local orthonormal basis $\{e_{\parallel},e_{\perp}^{(a)}\}$ such that  

$$
j^\mu = |j| e_{\parallel}^\mu ,\qquad 
e_{\perp}^{(a)}\cdot j=0,
$$

the metric matrix becomes block‑diagonal:

$$
g_{\mu\nu} =
\begin{pmatrix}
 g^{(0)}_{\parallel}+ \dfrac{\zeta_C}{\lambda_C C} |j|^2 & 0\\
 0 & g^{(0)}_{\perp}
\end{pmatrix}.
\qquad\text{(6.3)}
$$

The **parallel component** is therefore *inflated* relative to the perpendicular block.

#### Why distances *shrink* along the flow  

Geodesic length is obtained from the line element

$$
ds^2 = g_{\mu\nu} dx^\mu dx^\nu .
\qquad\text{(6.4)}
$$

In a Lagrangian formulation of particle motion on a Riemannian manifold the kinetic term reads  

$$
L = \frac12 g_{\mu\nu} \dot x^{\mu}\dot x^{\nu},
\qquad
\dot x^\mu:=\frac{dx^\mu}{ds}.
\qquad\text{(6.5)}
$$

Because the inverse metric $g^{\mu\nu}$ appears when solving for the physical velocity $\dot x^\mu$, an *increase* of the parallel component in $g_{\mu\nu}$ corresponds to a *decrease* of its inverse component $g^{\parallel\parallel}$.  Consequently, for a given amount of kinetic “energy’’ the particle can traverse a larger coordinate distance **along** the current than it could orthogonal to it.  In geometric terms the **geodesic between two points that are aligned with the flux is shorter** than the same Euclidean separation would suggest.

In short, a strong coherent current **reshapes the metric so that the shortest‑energy path (the geodesic) follows the flow**, turning the current into an intrinsic shortcut.

### 6.2 Physical picture: shortcuts along activity  

The mathematical effect described above admits a clear phenomenological interpretation:

* The coherence field $C$ carries **energy‑flow** in the form of a current $j^\mu=C v_C^\mu$.  
* Wherever that flow is intense, the geometry *adapts*: the metric stretches in the direction of the flow, which – because the geodesic cost uses the inverse metric – makes it **cheaper** (in terms of proper length) to move *with* the flow than across it.  

These three regimes give rise to a **hierarchy of “highways’’ inside the organism**:

* **Strong coherent current between basins.**  When a pronounced flux $j^\mu$ runs from basin $B_i$ toward basin $B_j$, the parallel component of the metric is amplified by the term $\zeta_C j_\mu j_\nu$ in (6.2).  As a result, the geodesic that follows this direction becomes shorter than it would be in the baseline geometry becing effectively an **energy‑flow shortcut**. The organism experiences a direct route along its own activity.
* **Region with negligible flux (or only density/gradient contributions).** In areas where $j^\mu$ is essentially zero, the extra $\zeta_C j_\mu j_\nu$ piece disappears from (6.2). Distances are then governed solely by the density term $\lambda_C C g_{\mu\nu}$ and the gradient‑pressure term $\xi_C\nabla_\mu C\,\nabla_\nu C$, providing a **neutral terrain** in which moving costs the baseline amount set by coherence density.
* **Sharp transverse gradients ($\xi_C$ large).** Where the coherence field varies steeply across a direction, the gradient term dominates.  This contracts the metric perpendicular to the gradient, causing geodesics to bend *away* from the steep wall creating an effective **barrier effect** that makes the system avoid crossing high‑gradient zones unless forced.

Because the current $j^\mu$ itself is generated by the reflexive loop ($C\!\to\!K\!\to\!g\!\to\!J_C\!\to\!C$), these shortcuts are **self‑reinforcing**. A pathway that already carries coherence becomes geometrically easier, encouraging even more flux to travel there.  The net result is an *anisotropic* distance network that constantly reshapes itself around the organism’s own activity patterns.

## 7 Hierarchical (Multi‑scale) Distances  

The Reflexive‑Coherence (RC) framework endows every **identity basin** $B_i(t)$ with its own geometry, and the geometry of the whole support $\Omega_t$ is obtained by stitching those pieces together.  Consequently the notion of “distance’’ naturally splits into three nested levels: microscopic, mesoscopic and macroscopic, each governed by a different proper‑time clock.  This hierarchy mirrors the **hierarchy of proper times** introduced in chapter 4.

### 7.1 Microscopic distances – inside a single basin  

Within a basin $B_i(t)$ the metric is simply the restriction  

$$
g^{(i)}_{\mu\nu}(x,t)=g_{\mu\nu}(x,t)\Big|_{x\in B_i(t)},
\qquad\text{(7.1)}
$$

and the associated proper‑time element reads

$$
d\tau_i = \sqrt{- g^{(i)}_{\mu\nu} dx^\mu dx^\nu}.
\qquad\text{(7.2)}
$$

Geodesic length between two points $x_1,x_2\in B_i(t)$ is obtained by extremising  

$$
L_{B_i}[\gamma]=\int_{\gamma}\sqrt{g^{(i)}_{\mu\nu} \dot x^\mu\dot x^\nu} ds,
\qquad\text{(7.3)}
$$

and the minimal value of $L_{B_i}$ is the **microscopic distance**. Because the metric inside a basin evolves only through the local coherence field, microscopic distances change smoothly as the basin’s internal density, gradient and flux are reshaped.

### 7.2 Mesoscopic distances – between neighboring basins  

When a path crosses from $B_i$ into an adjacent basin $B_j$ the line element switches from $g^{(i)}_{\mu\nu}$ to $g^{(j)}_{\mu\nu}$.  The total length is therefore a **piecewise‑geodesic** sum

$$
L_{ij}[\gamma]=
\int_{\gamma\cap B_i} \sqrt{g^{(i)}_{\mu\nu} \dot x^\mu\dot x^\nu} ds +
\int_{\gamma\cap B_j} \sqrt{g^{(j)}_{\mu\nu} \dot x^\mu\dot x^\nu} ds .
\qquad\text{(7.4)}
$$

Because each basin carries its own proper‑time clock (Eq. 7.2), the **mesoscopic distance** can be expressed in either of those clocks. Equivalently one may convert to a common reference (e.g. the global coordinate time) using the local relations  

$$
d\tau_i = \alpha_{ij} d\tau_j ,\qquad
\alpha_{ij}= \sqrt{\frac{-g^{(i)}_{\mu\nu}v^\mu v^\nu}{-g^{(j)}_{\rho\sigma}v^\rho v^\sigma}} ,
\qquad\text{(7.5)}
$$

where $v^\mu$ is the tangent to the crossing segment.  This conversion makes explicit the **inter‑basin “clock mismatch’’**.  In practice the mesoscopic distance is what an organism experiences when it moves from one functional module to a neighboring one. The motion feels smoother or more costly depending on how well the two local proper times align.

### 7.3 Macroscopic distances – across $\Omega_t$  

At the largest scale we ignore basin boundaries and treat the whole support as a single Riemannian manifold $(\Omega_t,g_{\mu\nu})$.  The **global geodesic** between two arbitrary points $x_1,x_2\in\Omega_t$ is defined by

$$
d_{\text{macro}}(x_1,x_2)=
\underset{\gamma: x_1\to x_2}{\min}
\int_{\gamma}\sqrt{g_{\mu\nu}(x,t) \dot x^\mu \dot x^\nu} ds .
\qquad\text{(7.6)}
$$

Because the metric itself is a functional of the **entire coherence field** (Eq. 5.1), macroscopic distances encode the cumulative effect of *all* basins, their internal fluxes and gradients, and any anisotropies introduced by strong currents.  As basins split or merge during refinement, the global geodesic network is continuously re‑wired. Consequently a pair of points that were once far apart may become much closer after a cascade of basin creations.

## 8 Fractal Extension (FRC) - Scale‑Expanding Hierarchies 

When the organism’s architecture itself **grows across scales** (e.g. a branching vascular network, a nested cognitive hierarchy, or a self‑similar colony), the single‑field Reflexive‑Coherence (RC) description must be enriched.  The **Fractal Reflexive Coherence (FRC)** formalism does exactly that: it adds an extra *scale coordinate* $\sigma\in[0,\infty)$ and promotes the coherence density to a **scale‑resolved field**  

$$
C\;=\;C(x,t;\sigma).
\qquad\text{(8.1)}
$$

All of the RC machinery (coherence tensor, metric, geodesics) is retained on each $\sigma$-slice, while a new *scale‑flux* couples the slices into an infinite cascade.

### 8.1 Scale coordinate and total coherence  

The **total coherence** of the whole organism, now a product manifold $\Omega_t\times[0,\infty)$, is defined as  

$$
C_{\mathrm{tot}}(t)=\int_{0}^{\infty} d\sigma \int_{\Omega_t} C(x,t;\sigma) dV_g ,
\qquad\text{(8.2)}
$$

and is **exactly conserved** under the FRC dynamics.  Thus no “new spatial volume’’ appears. Growth lives entirely in the extra $\sigma$-direction and in an ever‑more refined partition of the original support $\Omega_t$.

### 8.2 Scale‑flux and the cascade equation  

Coherence is transferred between neighboring scales by a **scale‑flux** $J^{\sigma}(x,t;\sigma)$.  The continuity equation acquires an extra term:

$$
\partial_t C + \nabla_{\mu} J_C^{\mu} + \partial_{\sigma} J^{\sigma}=0 .
\qquad\text{(8.3)}
$$

* The spatial flux $J_C^{\mu}=C\,v_C^{\mu}$ is the same as in ordinary RC.  
* The term $\partial_{\sigma}J^{\sigma}$ implements a **mass‑preserving cascade** from coarse to fine scales: when $J^{\sigma}>0$ coherence flows *down* the scale ladder, generating finer basins. When $J^{\sigma}<0$ it moves *up*, merging fine structures into coarser ones.  Because the integral (8.2) is invariant, the cascade merely **re‑distributes** coherence without loss.

### 8.3 Integrated coherence tensor and metric  

On each scale slice we can construct a local coherence tensor exactly as in RC:

$$
K_{\mu\nu}(x,t;\sigma) = \lambda_C\,C\,g_{\mu\nu} +\xi_C \nabla_\mu C \nabla_\nu C +\zeta_C j_\mu j_\nu , \qquad j_\mu=C v_{C,\mu}.
\qquad\text{(8.4)}
$$

The **scale‑integrated tensor** that governs the *global* geometry is then  

$$
K_{\mu\nu}^{\mathrm{(int)}}(x,t) = \int_{0}^{\infty} K_{\mu\nu}(x,t;\sigma) d\sigma .
\qquad\text{(8.5)}
$$

Finally we feed this integrated tensor into the same functional map used in RC:

$$
g_{\mu\nu}(x,t)=g_{\mu\nu} \left[ K^{\mathrm{(int)}}(x,t) \right] .
\qquad\text{(8.6)}
$$

Thus **the metric automatically incorporates contributions from *all* scales**, preserving the chain  

$$
C(x,t;\sigma) \longrightarrow  K_{\mu\nu}(x,t;\sigma) \longrightarrow  g_{\mu\nu}(x,t)
$$

that underlies every distance, curvature and proper time in the theory.

### 8.4 Self‑similar basins and identity tubes  

Each scale slice possesses its own set of **identity basins** $B_i(t;\sigma)$, the connected components of $\{C>0\}$ at that $\sigma$.  Because the cascade is *self‑similar* (the same continuity law holds on every slice), the basins form a **nested, tube‑like hierarchy** in the extended space $(x,\sigma)$:

$$
\mathcal T_i \;=\;\left\{\, (x,\sigma) \mid x\in B_i(t;\sigma)\,\right\}.
$$

These tubes are *multiscale attractors*. A collapse at any scale deepens the tube locally, while a split creates two daughter tubes that inherit the parent’s geometry.  The invariance $\Phi_t(A_\Sigma)=A_\Sigma$ guarantees that **identity persists** even as the cascade continuously spawns finer sub‑identities.

## 9 Summary & Outlook  

Below we bring together the three pillars that have been built throughout the paper and sketch a short agenda for future work.

### 9.1 Recap of the core results

- **Support and topology** The *emergent space* is the support $\displaystyle\Omega_t=\mathrm{supp}C(\cdot,t)$.  Topological changes are realised by **splits, merges or extinctions of basins** – i.e. by bifurcations of the zero‑set of $C$ – without ever adding new coordinate points.
- **Metric and distance** The scalar field generates a **coherence tensor** $K_{\mu\nu}[C]$. The metric is obtained as a functional $g_{\mu\nu}=g_{\mu\nu}[K]$.  Distance between any two points is the length of the geodesic that extremises $L[\gamma]=\int_\gamma \sqrt{g_{\mu\nu}\dot x^\mu\dot x^\nu} ds$.
- **Proper time** Restricting the metric to a basin $B_i(t)$ yields a local line element $d\tau_i=\sqrt{-g^{(i)}_{\mu\nu}dx^\mu dx^\nu}$.  Different basins therefore run on *different clocks*.
- **Spacetime “growth’’** Because the coordinate chart $[0,1]^d$ never expands, growth is realized as a **refinement of the coherence landscape**. New basins appear, existing ones deepen (larger curvature wells), and the metric reshapes accordingly.

In short, **distance, proper time and what we call “spacetime growth’’ are not independent primitives**; they are all *derived* from a single non‑negative scalar field $C$ via the chain  

$$
C \longrightarrow K_{\mu\nu}[C] \;\longrightarrow\; g_{\mu\nu} \longrightarrow
\begin{cases}
\text{geodesic length (distance)}\\
\text{proper‑time element }d\tau_i
\end{cases}.
$$

### 9.2 Why “growth’’ means *refinement*  

* The **ambient coordinate set** never changes; $\Omega_t$ is always a subset of the fixed chart.  
* **Refinement** occurs when the scalar field’s Hessian develops new critical points, thereby **splitting a basin into two children** or **deepening an existing basin**.  The metric responds instantly because it is a functional of $C$.  
* In the **Fractal Reflexive Coherence** extension the same idea is pushed to an infinite hierarchy by introducing a scale coordinate $\sigma$. Growth then lives entirely in the additional $\sigma$-direction while the spatial support remains unchanged.

Thus “spacetime expansion’’ in RC is best understood as **the emergence of richer internal geometry**, not as the addition of new points to space.

### 9.3 Concluding remark  

The Reflexive‑Coherence framework demonstrates that **a single scalar coherence field suffices to generate an entire spacetime‑like geometry**, complete with distance, proper time and a notion of growth that is purely *refinitional*.  By treating topology change as a smooth redistribution of the same conserved quantity, RC sidesteps the need for ad‑hoc surgical operations on manifolds while still reproducing many phenomenologically familiar effects (gravity‑like attraction, anisotropic shortcuts, fractal hierarchies).  

