# The Quest for Stable RC PDEs

**Part I – Constructing and Debugging a Coherence-based Simulator**

## Abstract

We report on a series of numerical experiments aimed at constructing a stable partial-differential implementation of the Reflexive Coherence (RC) dynamics. Rather than presenting a polished final model, we document the full trajectory: the initial choices of fields and potentials, the discretization strategy, the failure modes (single-basin collapse, runaway coarsening, numerical blow-up), and the removal of ad-hoc operators such as explicit “collapse” rules. The experiments are carried out on a 2D grid (typically $128\times 128$ with spacing $\Delta x = \Delta y = 0.1$) with explicit Euler timestepping. We show how specific parameter choices—e.g. $\lambda = 0.8$, $\xi = 0.1$, $\kappa = 0.05$, $\alpha = 1.0$, $\beta = 0.1$, $\gamma_{\text{curv}} \approx 0.3$—lead to qualitatively different regimes, and we give concrete examples of both stable and pathological runs. The goal is didactic: a reconstruction guide for RC simulations that makes visible the “engineering” layer between the abstract equations and usable code.

## 1. Introduction

The Reflexive Coherence (RC) formulation describes identity as a dynamical configuration of a coherence field and its flux. In the continuum picture, this is expressed as coupled PDEs for a scalar coherence density (C(x,t)), a coherence flux (J_C(x,t)), and, in full ROM, additional reserve and compatibility fields.

This paper is **not** about the final, “correct” RC equations.
Instead, it is about the **process** of getting any nontrivial RC-like PDE to run stably enough that its phenomenology can be studied.

We deliberately keep the narrative empirical:

* we start from a minimal scheme that *should* support multiple basins,
* show why it first collapses to a single, central identity,
* document attempts to “help” the system via explicit collapse and spark operators,
* and end by removing these shortcuts and clarifying what is inherently stable in the PDE, and what must be delegated to higher levels (e.g. analysis rather than direct operators).

Throughout, we use **concrete parameter values** and report the actual behaviors observed, so that readers can reproduce and extend the experiments.

## 2. Numerical Setup

### 2.1 Fields and domain

We work on a rectangular grid $\Omega \subset \mathbb{R}^2$ discretized as

$$
\Omega = { (i\Delta x, j\Delta y) \mid i = 0,\dots,n_x-1,; j=0,\dots,n_y-1 }
$$

with typical choices

* $n_x = n_y = 128$,
* $\Delta x = \Delta y = 0.1$.

The primary field is the **coherence density**

$$
C_{ij}(t) \approx C(i\Delta x, j\Delta y, t) \ge 0.
$$

In “RC mode” we also evolve a reserve-like field $R_{ij}(t))$ and interpret the combination as a reduced RC system. Where necessary, we denote the coherence flux as

$$
J_C = C v_C,
$$

but in the present implementation this appears implicitly through the PDE form, not as a separate stored field.

### 2.2 Schematic form of the PDE

The actual code evolved $C$ using an explicit Euler step

$$
C^{(n+1)} = C^{(n)} + \Delta t \mathcal{L}[C^{(n)}; \theta],
$$

where $\mathcal{L}$ is a spatial operator depending on parameters $\theta$. In schematized form,

$$
\mathcal{L}[C] = \underbrace{\lambda \Delta C}_{\text{diffusion}} - \underbrace{\alpha \partial_C V(C)}_{\text{local potential}} + \underbrace{\kappa \mathcal{L}_{\text{curv}}[C;\gamma_{\text{curv}}]}_{\text{curvature / pattern term}} + \underbrace{\xi \mathcal{N}[C]}_{\text{nonlinear RC-like term}}
$$

with:

* $\lambda$ — diffusion coefficient;
* $V(C)$ — a potential, either double-well or plateau;
* $\kappa, \gamma_{\text{curv}}$ — curvature coupling and associated shape parameter;
* $\xi$ — strength of an additional nonlinear term (e.g. reserve coupling in RC mode).

We used **explicit finite differences** for:

* the Laplacian $\Delta C$ (5-point stencil),
* first derivatives (2-point central differences),
* and higher derivatives where needed (e.g. for curvature-like operators).

### 2.3 Time stepping

We experimented with two orders of magnitude for the timestep:

* A **very small step**

  $$
  \Delta t = 10^{-5}, \quad \text{with } \texttt{steps} = 500,
  $$
  in a parameter set
  $$
  (n_x,n_y)=(128,128), \Delta x = 0.1, \lambda=0.8, \xi=0.1, \zeta=0.05, \kappa=0.05, \alpha=1.0, \beta=0.1.
  $$

* A **larger step**

  $$
  \Delta t = 10^{-3}, \quad \texttt{steps} = 100,
  $$
  in runs focussed on short-time behavior:
  $$
  \lambda = 1.0, \xi = 0.8, \kappa = 0.05, \alpha = 0.5, \beta = 0.8, \gamma_{\text{curv}} = 0.3.
  $$

In all cases we used simple explicit Euler updates; the small-$\Delta t$ configuration was chosen primarily for robustness rather than efficiency.

### 2.4 Potentials: double-well vs. plateau

We implemented two kinds of local potential:

1. **Double-well**: encourages $C$ to localize near two preferred coherence levels (e.g. “empty” vs “occupied”).
2. **Plateau**: a nearly flat region of low energy between $C_{\text{low}}$ and $C_{\text{high}}$, with steeper walls below / above:

   ```json
   "potential": "plateau",
   "potential_params": {
     "C_low": 0.05,
     "C_high": 0.6,
     "k_low": 0.2,
     "k_high": 0.2
   }
   ```

Despite their conceptual differences, we observed that under some parameter regimes the early-time dynamics of these two potentials are nearly indistinguishable; this becomes an important diagnostic in §4.

## 3. First Attempts and Pathologies

### 3.1 Initial condition: many basins, one survivor

A natural initial experiment was to:

* seed multiple coherence basins (e.g. $\texttt{basins} = 100$) as localized bumps,
* evolve with the parameter set

  $$
  \lambda=0.8, \xi=0.1, \zeta=0.05, \kappa=0.05, \alpha=1.0, \beta=0.1,
  \Delta t = 10^{-5}, \texttt{steps} = 500,
  $$

* and visually inspect the evolving field via an animated $128\times 128$ (or $512\times 512$) heatmap.

Despite the large number of initial basins, the typical outcome was:

* rapid diffusion and interaction,
* coarsening of the pattern,
* and **eventual dominance of a single large basin near the center of the domain**.

The precise location varied, but with symmetric boundary conditions and potentials, a central attractor was the most common result. This behavior is perfectly consistent with reaction–diffusion intuition, but it **contradicts the RC intuition** where multiple persistent identities are natural rather than fine-tuned.

### 3.2 Tuning curvature and RC parameters

We then explored the roles of:

* curvature strength $\kappa$,
* curvature shape parameter $\gamma_{\text{curv}}$,
* nonlinear coupling strengths $\xi, \beta$,

to see if they could stabilize multiple basins.

For example, using

$$
\begin{multline}
\lambda = 1.0, \xi = 0.8, \kappa = 0.05, \alpha = 0.5, \beta = 0.8, \gamma_{\text{curv}} = 0.3, \Delta t = 10^{-3}, \\ \texttt{steps} = 100, \texttt{basins} = 4,
\end{multline}
$$

we observed:

* faster pattern formation: filaments and proto-“cosmic web” structures appeared around $t \approx 30\Delta t$,
* but still a strong tendency toward **coarsening**: smaller basins were gradually absorbed by larger ones.

Increasing $\kappa$ to $0.1$ in otherwise similar runs sharpened the emergent network, but did not prevent eventual dominance by a small number of basins. Conversely, reducing $\kappa$ to $0.01$ suppressed filament-like structures and reverted the system to smoother coarsening.

### 3.3 A naive collapse operator

To artificially prevent multiple nearby maxima from coexisting, we added a “collapse” operator acting directly on $C$. In code, this took the form:

* find all local maxima of $C$,
* cluster maxima closer than a distance $d_{\text{collapse}}$,
* keep only the highest maximum in each cluster,
* set a small radius around discarded maxima to a low background level.

Concretely, we used:

* grid distance in index units (not physical units),
* values $d_{\text{collapse}} = 0.8, 1.5$ in separate tests,
* background level set to 1% of the current mean coherence:
  $$
  C_{\text{bg}} = 0.01 \cdot \langle C \rangle.
  $$

This operator turned out to be problematic in several ways:

1. **Grid-geometry artifact**.
   On an integer grid, the squared distance between distinct cells is at least 1.
   Thus for $d_{\text{collapse}} = 0.8$, no pair of distinct maxima ever satisfied
   $(\Delta i)^2 + (\Delta j)^2 \le d_{\text{collapse}}^2$, so collapse never triggered.

2. **Non-conservation and surgery.**
   When $d_{\text{collapse}} \ge 1$, collapse would aggressively flatten disks around discarded maxima, destroying coherence mass without redistributing it. This violates any conservation or Lyapunov structure we might want from RC.

3. **Conceptual mismatch.**
   In the RC picture, “collapse” is a **decision of the flow**—which basin the flux commits to—not a direct rewrite of geometry. Geometry should follow from sustained flow, not from an explicit field surgery.

In later runs we discovered that with **rc_mode = true** the code path had effectively disabled collapse entirely, so the “helpful” operator had in fact not been active in the experiments discussed in §3.1–3.2. Nevertheless, the attempt clarified the conceptual error: collapse should not be coded as a local field modification at all.

### 3.4 A spark operator that fires continuously and keeps re-fragmenting basins

In addition to collapse, we introduced a “spark” mechanism intended to model the emergence of a new basin when the coherence field becomes locally critical. The triggering condition was inspired by the criterion

$$
\det \mathrm{Hess}(C) \approx 0
$$

at points where $\nabla C \approx 0$, indicating a near-degenerate critical point.

In practice, the initial implementation:

* evaluated gradient and Hessian on the entire grid,
* used fixed thresholds (e.g. `spark_epsilon = 1e-6`) that did **not** scale with typical gradient or curvature magnitudes,
* triggered a spark whenever the condition was satisfied at *any* grid point.

Numerical tracing showed that:

* **sparks were firing on essentially every “event” step**, often at multiple locations,
* the additional coherence injected by a spark did not stay local: if another denser basin was present, the sparked coherence typically **flowed into that denser basin** along preferred paths,
* individual basins were repeatedly **split and reshaped**: a previously dominant basin could fracture into several maxima, with the dominant peak relocating to a nearby denser point.

From a stability-engineering perspective, this meant that the spark operator was acting as a **strong, time-dependent nonlocal forcing**:

* the base PDE dynamics (diffusion + potential + curvature + RC coupling) never had a chance to relax toward their own attractor,
* the system remained in a state of ongoing fragmentation and re-coarsening driven by the spark rule, rather than by the intrinsic RC-like terms.

As with collapse, this made it difficult to disentangle what was **inherently stable in the PDE** from what was an artifact of the control layer. This motivates the later decision (cf. §4.3 and §5) to treat sparks as *diagnostic events* inferred from the dynamics, rather than as primitive operators that directly modify the field.

## 4. Lessons About Stability from These Failures

The experiments above, while “wrong” in terms of intended phenomenology, were informative about **what not to do** when building RC-like PDEs.

### 4.1 Single-basin dominance is generic in naive reaction–diffusion

Given:

* symmetric domain and boundary conditions,
* symmetric potentials (double-well or plateau),
* diffusion and attractive nonlinearities,

the **generic** outcome is:

* coarsening,
* annihilation of smaller domains,
* convergence to one or a few large basins.

This is standard in Allen–Cahn and Cahn–Hilliard-type systems. Our runs confirmed that simply adding extra terms with parameters $\xi,\kappa,\gamma_{\text{curv}}$ does not automatically break this tendency.

**Implication:**
RC-like multiplicity of identities cannot be taken for granted; it must be *engineered* into the functional, not merely hoped for via random extra terms.

### 4.2 Explicit collapse operators are conceptually and numerically suspect

The naive collapse operator taught three distinct lessons:

1. Field-surgery rules (flatten nearby basins) **break conservation** and any hope of a clean variational formulation.
2. Discrete grid geometry can make seemingly “intuitive” distances meaningless (e.g. $d_{\text{collapse}} = 0.8$ doing nothing).
3. Most importantly: in RC, **collapse should be an emergent property of trajectories**, not a direct operator.

In subsequent iterations, all collapse-related code was removed from the core dynamics. The intended use of “collapse” became observational: a label for when the number of effective basins decreases, not a primitive action.

### 4.3 Sparks must be rare, relative, and tied to mismatch

The behavior of the initial spark implementation—firing on virtually every event step and continuously reshaping basins—demonstrated several pitfalls:

* fixed absolute thresholds (e.g. `det_eps = 1e-6`) are inappropriate when typical $|\det \mathrm{Hess}(C)|$ is of order $10^3$–$10^4$ for $\Delta x = 0.1$;
* a global “if any point qualifies, spark” logic guarantees **ongoing activation** somewhere in the domain;
* the added coherence does not remain local: it often **flows into other, denser basins**, so sparks effectively act as a strong time-dependent forcing that drives cycles of fragmentation, migration, and re-coarsening.

In other words, sparks in this form do not merely “add a bit of structure”—they dominate the dynamics and make it hard to assess the intrinsic stability of the underlying PDE.

A more realistic design must therefore:

* define thresholds **relative** to the current field:

  * $|\nabla C| < \varepsilon_{\text{grad}}$ where $\varepsilon_{\text{grad}}$ scales with a median or maximum gradient,
  * $|\det \mathrm{Hess}(C)| < \varepsilon_{\text{det}}$ where $\varepsilon_{\text{det}}$ scales with typical curvature on near-critical points;

* and include **cooldown** mechanisms so the same location cannot trigger sparks at every event step.

Even with better thresholds, the experiments suggest that sparks, like collapses, are better treated as **measured phenomena** (e.g. births of new basins or fission events inferred from the evolving landscape) rather than as primitive operators. For the purpose of building and understanding a stable RC-like PDE, it is cleaner to:

1. run the PDE without any explicit spark rule,
2. and then *identify* “spark events” retrospectively as times when new basins appear, split, or redirect flow, rather than forcing those events in the evolution equation itself.

## 5. A Minimal Reconstruction Protocol

Based on these experiments, we propose the following protocol for anyone attempting to reconstruct RC-like PDE dynamics.

### Step 1 – Fix the grid and a conservative baseline

Start with:

* $n_x = n_y = 128$,
* $\Delta x = 0.1$,
* explicit Euler with

  $$
  \Delta t = 10^{-5}
  $$

  for robustness.

Implement only:

$$
\partial_t C = \lambda \Delta C - \alpha \partial_C V(C)
$$

with a simple double-well or plateau potential and parameters in the range:

* $\lambda \in [0.5, 1.0]$,
* $\alpha \in [0.5, 1.0]$.

Seed a small number of basins (e.g. $\texttt{basins} = 4$) and verify:

* the code runs without blow-up for 500–1000 steps,
* patterns coarsen in the expected way,
* numerical artifacts (checkerboarding, ringing) are absent.

### Step 2 – Add curvature / pattern terms gradually

Introduce a curvature-related term parameterized by $\kappa$ and $\gamma_{\text{curv}}$:

* start with $\kappa = 0.01$, $\gamma_{\text{curv}} = 0.3$,
* then increase (\kappa) to (0.05) and (0.1) to observe:

  * sharpening of interfaces,
  * appearance of filamentary patterns.

At this stage, **do not** include sparks or collapse. The goal is to see how far a purely local PDE can go toward nontrivial structure.

### Step 3 – Enable RC-mode coupling

Introduce an additional field $R(x,t)$ and a coupling term controlled by $\xi$, $\beta$, etc., for instance:

* $\xi \in [0.1, 0.8]$,
* $\beta \in [0.1, 0.8]$.

Again, test small and large values:

* for $\xi = 0.1$, the effect should be a mild modulation of growth;
* for $\xi = 0.8$, strong feedback can produce web-like structures, but may also destabilize the scheme if $\Delta t$ is too large.

Confirm that:

* no new pathologies arise (e.g. negative coherence, unbounded growth),
* coarsening behavior is modified but not dominated by numerical noise.

### Step 4 – Treat sparks and collapses as diagnostics, not operators

Rather than coding `apply_spark` and `apply_collapse` into the dynamics:

1. Run the PDE with the chosen parameters (e.g. $\lambda=1.0, \xi=0.8, \kappa=0.05, \gamma_{\text{curv}}=0.3, \Delta t = 10^{-3}, \texttt{steps} = 1000$).

2. At each frame, **analyze**:

   * number of local maxima,
   * clustering of maxima within a physical radius,
   * birth and death of basins across time.

3. Define:

   * a *spark event* as an increase in the number of basins, or the appearance of a new maximum above some fraction of the global mean,
   * a *collapse event* as a decrease in the number of basins or the merging of two maxima within a given time window.

In this way, the PDE remains purely local and conservative (modulo known reaction terms), and RC notions like sparks and collapse are **read off** from the behavior instead of injected as external forces.

## 6. Discussion and Outlook

The experiments reported here started from an ambition that sounds simple: “simulate RC as a PDE for coherence.” What we learned is that even before debating the exact form of the RC equations, one must solve a more mundane problem: **build a numerically stable, conceptually clean PDE that does not secretly implement unwanted assumptions.**

The main lessons are:

1. **Naive reaction–diffusion tends to a single identity.**
   Without carefully designed pattern-forming or saturation terms, multiple identities are not stable.

2. **Explicit control operators obscure the dynamics.**
   Collapse and spark operators, when encoded as direct modifications of the field, can mask underlying behavior, break conservation, and diverge from the theoretical picture.

3. **Diagnostics are better than hacks.**
   Treat sparks and collapse as *labels* for what the PDE does, not as imperative instructions to the PDE.

4. **Concrete parameter sweeps are essential.**
   Being explicit about values such as $\lambda = 0.8$, $\xi = 0.1$, $\kappa = 0.05$, $\Delta t = 10^{-5}$ vs $10^{-3}$ is not mere implementation detail; it is part of understanding the phenomenology of RC-like fields.

In **Part II**, we will turn away from the “quest for stability” and focus instead on **observing and classifying the behavior** of the resulting systems: how filaments, webs, and basins appear; how identity basins interact; and how one can define and measure sparks and collapses as emergent phenomenology rather than code-level constructs.
