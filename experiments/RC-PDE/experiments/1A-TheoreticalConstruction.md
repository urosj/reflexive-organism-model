# The Quest for Stable PDEs in Reflexive Coherence Dynamics

## Abstract

We report on the practical construction of a numerically stable partial-differential realization of Reflexive Coherence (RC). Starting from the conceptual ROM–RC picture—coherence as a scalar field, curvature-induced basins as identities, and sparks as topological changes—we attempted a direct PDE implementation with local diffusion, nonlinear potentials, curvature terms, and event-driven “spark” and “collapse” operators. The resulting dynamics repeatedly exhibited pathological behavior: spontaneous blow-up, negative coherence, artificial annihilation of basins, and a tendency toward a single monolithic attractor. Through successive constraints—positivity preservation, boundedness, step-size control, removal of “hard” collapse, and careful redefinition of sparks—we arrive at a set of design principles for stable RC PDEs. Rather than prescribing a final equation, this paper documents the *method* by which one can approach RC-style PDEs without violating the theory’s invariants or being misled by numerical artifacts.

## 1. Introduction

Reflexive Organism Models (ROM) and their refinement as Reflexive Coherence (RC) describe identity as a dynamical invariant emerging from the interaction of several fields: structure, reserve, compatibility, and experience. In the coherence-only reduction, these are compressed into a single scalar field $C(x,t)$ and associated geometry: basins of $C$ act as identities, while curvature and flux encode their interaction, reproduction, and pruning.

The conceptual picture is clear: a coherence landscape with multiple basins, sparks at degeneracies of the Hessian, and collapses when flux commits to one basin over competitors. What is *not* obvious is how to implement this in a concrete PDE simulation that is both numerically stable and faithful to the theory’s invariants.

This paper is deliberately narrow. We do **not** ask what an RC system does once it is well-posed. Instead, we ask:

> *How do we construct a PDE system that is stable enough to be trusted as an implementation of RC, rather than an artifact of the numerics?*

The answer came not from a single “correct PDE” but from a sequence of failures and corrections. We report those steps here so that future reconstructions of RC can avoid repeating the same mistakes.

A companion paper will focus on the phenomenology of the resulting systems once stability is secured.

## 2. From Conceptual RC to a PDE Ansatz

### 2.1 Coherence field and geometry

We work on a fixed spatial domain $\Omega \subset \mathbb{R}^2$ discretized as a regular grid of spacing $\Delta x$. The primary field is the **coherence density**

$$
C(x,t) \ge 0, \quad x \in \Omega, t \ge 0,
$$

optionally accompanied by a reserve field (R(x,t)) and a coherence-induced geometry tensor $K_{\mu\nu}[C]$, as in the coherence-only ROM.

The generic PDE ansatz for coherence evolution took the form

$$
\partial_t C = F[C] \equiv
D \Delta C - \frac{\partial V(C)}{\partial C} + \mathcal{K}[C] + \mathcal{N}[C, R] ,
$$

where

* $D$ is a diffusion coefficient,
* $V(C)$ is an effective potential (double-well or plateau),
* $\mathcal{K}[C]$ encodes curvature-driven terms (e.g. $\xi \Delta^2 C$ or gradient nonlinearities),
* $\mathcal{N}[C,R]$ contains RC-specific couplings (growth, saturation, reserve feedback).

We used a family of parameters $\lambda, \xi, \zeta, \kappa, \alpha, \beta, \gamma_{\text{curv}}$ to modulate these contributions, but, importantly, **no particular choice of coefficients saved us from instability** until we enforced structural constraints on the scheme itself.

### 2.2 Identity basins and events

Identity basins were defined as **local maxima** of $C$, detected on the grid by a discrete “find local maxima” operator. This choice makes basins a purely geometric notion, consistent with the static picture in ROM: an identity is a region of coherence bounded by separatrices.

Two special types of events were intended:

* **Sparks**: the emergence of a new basin when the landscape undergoes a local degeneracy, heuristically associated with $\det (\mathrm{Hess}(C)) \approx 0$.
* **Collapses**: resolution of competing basins when coherence flux commits to one and starves the others.

In early implementations, both events were realized as *explicit operators* acting on the field $C$. This decision turned out to be the main source of conceptual and numerical trouble.

## 3. Stability Requirements

Our first attempt at RC PDEs quickly revealed that the theory’s invariants must be translated into explicit **numerical constraints**. We found that a usable RC simulation must, at minimum, satisfy:

1. **Positivity**
   Coherence density must remain non-negative:
   $$
   C(x,t) \ge 0 \quad \forall x,t.
   $$
   Negative coherence has no meaning in the ROM picture and is purely a numerical artifact.

2. **Boundedness**
   Coherence should remain within an application-dependent upper bound $C_{\max}$, either enforced by the potential $V(C)$ or by saturation terms. Unbounded growth from nonlinear feedback is common in unconstrained PDEs and destroys the notion of stable basins.

3. **Controlled time stepping**
   For explicit schemes, the usual CFL-type constraints apply. For a simple diffusion term,
   $$
   \Delta t \lesssim \frac{\Delta x^2}{4D},
   $$
   and higher-order curvature terms $(\Delta^2 C)$ impose even stricter step-size limits. Many early “instabilities” were simply violations of these constraints.

4. **No hidden mass sources or sinks**
   Even when coherence itself is not formally conserved, any additional event operators (sparks, collapses) must not lead to unbounded global gain or loss of $C$ unless that is explicitly intended. Otherwise, one confuses numerical drift with genuine RC behavior.

5. **Topology changes must be emergent**
   RC treats sparks and collapses as *phenomena*: changes in the structure of basins and flux. Implementing them as arbitrary, non-local edits of the field easily violates stability and the conceptual model simultaneously.

Sections 4–6 recount how we violated these conditions, and what each failure taught us.

## 4. Early Failure Modes

### 4.1 Blow-up and negative coherence

Our initial PDE prototypes combined diffusion, a potential $V(C)$, and curvature-enhancing terms. Without strict bounds on $\Delta t$ and with naively chosen coefficients, we observed:

* Exponential blow-up of $C$ near maxima,
* Negative overshoots near steep gradients,
* Rapid oscillations at the grid scale.

These behaviors were insensitive to the particular RC parameters but highly sensitive to the discretization. This was the first indication that **numerical stability must be addressed before any interpretation of the dynamics**.

The fix was conventional: enforce a strict time-step bound, add a post-update projection $C \leftarrow \max(C, 0)$, and shape $V(C)$ so that $\partial V / \partial C$ grows rapidly for large $C$.

### 4.2 Ghost basins and grid artifacts

Even with bounded $C$, the combination of curvature terms and discrete derivatives produced spurious local maxima—“ghost basins”—that appeared and disappeared at the grid scale. These were visually convincing but physically meaningless.

We adopted two rules:

1. **Smoothing before detection**
   Local maxima are detected on a slightly smoothed version of $C$, e.g. after a small Gaussian blur, to suppress grid-scale ripples.

2. **Minimum basin size**
   Maxima that do not extend beyond a small neighborhood or carry negligible mass are ignored.

This reduced false positives but did not solve the deeper problem: how to let RC dynamics create and destroy basins without explicit intervention.

## 5. The Collapse Operator: A Useful Anti-Pattern

### 5.1 The naive collapse implementation

To enforce the idea that “nearby identities collapse into a single basin,” we introduced an explicit **collapse operator**:

```python
def apply_collapse(field: CoherenceField, d_collapse: float = 1.5) -> bool:
    """
    Detect and collapse nearby maxima into the strongest basin.
    If multiple maxima lie within d_collapse (grid units),
    only the highest is kept; other basins are reset to a background value.
    """
    y_idx, x_idx = find_local_maxima(field.C)
    maxima = list(zip(y_idx, x_idx, field.C[y_idx, x_idx]))
    if len(maxima) <= 1:
        return False

    maxima_sorted = sorted(maxima, key=lambda t: t[2], reverse=True)
    keep, discarded = [], []

    for y, x, val in maxima_sorted:
        if any((y - ky)**2 + (x - kx)**2 <= d_collapse**2 for ky, kx, _ in keep):
            discarded.append((y, x, val))
        else:
            keep.append((y, x, val))

    if not discarded:
        return False

    background = 0.01 * float(np.mean(field.C))
    for y, x, _ in discarded:
        # Suppress the basin around the discarded maximum
        r2 = (Y - y)**2 + (X - x)**2
        mask = r2 <= d_collapse**2
        field.C[mask] = background

    for y, x, val in keep:
        field.C[y, x] = val

    field.clip_nonnegative()
    return True
```

Conceptually, this procedure:

* Clusters local maxima within a radius $d_{\text{collapse}}$,
* Keeps only the tallest peak in each cluster,
* Flattens the neighborhood of the others to a small “background” level.

At first glance, this appears to implement the desired collapse of nearby basins into a single identity.

### 5.2 Conceptual violations

Closer analysis revealed several conflicts with RC:

1. **Collapse is not geometric surgery**
   In RC, collapse is a statement about **flux**: coherence trajectories commit to one basin, and competing basins are starved. Geometry changes *after* this decision, through the dynamics. Directly rewriting $C$ short-circuits the mechanism and confuses cause with effect.

2. **Non-conservation and hidden dissipation**
   Setting entire neighborhoods to a low background value unavoidably destroys coherence mass. Unless this loss is explicitly accounted for, one can no longer tell whether a basin disappeared due to RC dynamics or due to the operator.

3. **Arbitrary interaction radius**
   The radius $d_{\text{collapse}}$ is expressed in **grid units**, not physical distance. Moreover, when $d_{\text{collapse}} < 1$, no two distinct grid maxima can ever collapse, because the squared distance between distinct grid points is at least $1$. In this parameter regime the operator silently does nothing, giving a false sense of safety.

4. **Destruction of basin shape**
   Flattening everything within a disk around a discarded maximum also modifies the shape of the kept basins if disks overlap. Reinstituting only the peak value at the kept maximum leaves an artificial spike surrounded by a flattened region. This has no interpretation in RC.

### 5.3 Practical irrelevance and removal

A later trace revealed that, in the RC mode used for most experiments, the collapse operator was in fact **disabled**. No collapses were ever applied; the system’s behavior was entirely governed by the PDE and the spark operator.

This made it easy to take the decisive step: *remove all collapse-related code*. Collapse is now treated purely as an emergent phenomenon—loss of a basin in the measured landscape—rather than as an intervention.

The lesson for reconstructing RC is clear:

> **Do not implement collapse as a field-editing operator.**
> At most, detect collapse events as changes in the basin structure and optionally feed them back through *flux* bias, not geometric surgery.

### 6. Sparks: When Geometry Alone Misleads

#### 6.1 Geometric spark criterion

Sparks were intended to represent the birth of new identities when coherence cannot be reconciled with the existing landscape, conceptually tied to degeneracies in the Hessian of $C$: $\det(\mathrm{Hess}(C)) = 0$.

The implementation followed this logic:

1. Compute gradient $\nabla C$ and Hessian components $C_{xx}, C_{yy}, C_{xy}$.
2. Identify “almost critical” points where $\lVert\nabla C\rVert < \varepsilon_{\text{grad}}$.
3. Among those, identify “almost degenerate” points where $|\det(\mathrm{Hess}(C))| < \varepsilon_{\text{det}}$.
4. Use these points as candidates for spark insertion: small, localized perturbations of $C$.

In practice, we used fixed thresholds (e.g. `spark_epsilon = 1e-6`) and evaluated the conditions on the entire grid at designated “event steps”.

#### 6.2 Observed behavior: continuous firing, fragmentation, and redirected flow

With this rule in place and collapse disabled, the spark operator exhibited three robust behaviors:

1. **Continuous activation**
   Because $\varepsilon_{\text{grad}}$ and $\varepsilon_{\text{det}}$ were fixed in absolute units and not scaled to the typical gradients and curvatures in the field, there was *almost always* at least one grid point satisfying the criterion. As a result, sparks fired on essentially every event step, often at multiple locations.

2. **Repeated basin fragmentation**
   Sparks did not merely add a little mass to existing basins. When a spark occurred inside or near a basin:

   * that basin frequently **split into multiple local maxima**,
   * the dominant peak could **shift** to a nearby, denser location,
   * new secondary basins emerged around the original one.
     In other words, sparks acted as a **fission mechanism** that kept re-introducing multiplicity even after the PDE had coarsened toward one or a few strong basins.

3. **Flow into denser basins along preferred paths**
   The additional coherence injected by a spark did not remain localized. If another denser basin existed:

   * the sparked coherence tended to **drift into the denser basin**,
   * the drift followed **nontrivial paths**, often along filamentary regions created by curvature and RC coupling.
     This reveals an **effective, field-dependent curvature**: the combination of diffusion, nonlinearities, and curvature terms defines a landscape in which sparked coherence “falls” toward stronger identities rather than diffusing isotropically.

From a purely dynamical point of view, this made the system more interesting: basins fragmented, moved, and merged instead of freezing into a single static attractor. From the stability-engineering point of view, however, it meant that the spark operator was acting as a strong, persistent, nonlocal forcing on top of the PDE.

#### 6.3 Why this is still the wrong way to implement sparks

Even though the spark mechanism did *not* simply feed a single monolithic basin, it still violated several of the stability and conceptual requirements in §3:

1. **Geometry-only triggering**
   The spark rule relied solely on local differential geometry (gradients and Hessians of (C)). It did not encode any notion of:

   * prediction vs observation mismatch,
   * compatibility or incompatibility,
   * reserve imbalance.
     In ROM/RC terms, it was blind to *why* a new identity should appear; it only knew that the field looked “critical” in a local geometric sense.

2. **Uncontrolled, dense event rate**
   Because thresholds were absolute and checked globally, sparks fired almost every event step and at many sites. This kept the system in a perpetual state of externally enforced fragmentation and re-coarsening, making it hard to distinguish:

   * behaviors intrinsic to the PDE, from
   * behaviors imposed by the spark rule.

3. **Strong, time-dependent forcing**
   Each spark injected coherence and thereby modified the global mass distribution and the effective potential landscape. Frequent sparks meant the system was constantly being kicked by a nonlocal, time-dependent source term. This is the opposite of what we want when assessing **stability**: a stable PDE should be analyzable without being continually driven by large external events.

4. **Conceptual mismatch with RC sparks**
   In RC, sparks are not just “geometric oddities”; they are tied to *insufficiency* of the current identity configuration to account for ongoing interactions. A purely Hessian-based criterion cannot capture this. At best, geometry can indicate *where* a spark could occur; it cannot decide *whether* one should, in the RC sense.

These issues led to the same conclusion as with collapse: for the purpose of building a trustworthy RC-style PDE, **sparks should not be implemented as a hard operator that directly edits (C)**. Instead:

* run the PDE with no explicit spark rule,
* instrument the dynamics,
* and then *identify* spark-like events retrospectively (e.g. sudden appearance of new basins, fragmentation of existing ones, or redirection of flow).

Only in a later, more theory-driven model—where mismatch and compatibility are explicit—does it make sense to reintroduce sparks as genuine, state-dependent events rather than as geometry-only triggers.

## 7. Instrumentation and Diagnostics

A recurring theme in the quest for stability was the need for **instrumentation**: without explicit diagnostics, it is easy to misinterpret numerical accidents as RC phenomena.

The following diagnostics proved essential:

1. **Event tracing**
   At each “event step” we logged:

   * whether a spark fired,
   * whether a collapse (while it existed) was applied,
   * how many basins were present.

   This immediately revealed, for example, that sparks were firing at every event step and collapses never.

2. **Basin statistics over time**
   Counting local maxima (after smoothing) and grouping nearby ones allowed us to track:

   * number of effective basins,
   * their locations,
   * their integrated mass.

   Simple plots of basin count vs time are particularly revealing: coarsening to one, persistent multiplicity, or oscillations.

3. **Global invariants**
   Even when coherence is not formally conserved, monitoring:
   $$
   M_C(t) = \int_\Omega C(x,t),dx
   $$
   provides an immediate check on the net effect of event operators and potential terms. Unexplained drift in $M_C$ is almost always a bug.

4. **Scale-aware thresholds**
   Parameters like $\varepsilon_{\text{grad}}) and (\varepsilon_{\text{det}}$ for spark detection must be chosen relative to the actual distribution of gradients and Hessians in the field. Hard-coded values (e.g. $10^{-6}$) are meaningless without reference to the discretization.

Together, these tools turned the simulation from a black box into a system we could interrogate and debug.

## 8. Lessons for Reconstructing RC PDEs

The experiments described above suggest several general guidelines for anyone attempting to reconstruct RC as a PDE system:

1. **Stabilize first, interpret later**
   Ensure positivity, boundedness, and numerical stability *before* trying to interpret any emergent structure as “identity,” “spark,” or “collapse.”

2. **Avoid field-editing event operators**
   Do not implement conceptual events (collapse, spark) as direct modifications of (C(x,t)). Instead, let them be **labels** for what the dynamics have done, or, at most, gentle biases on the flux.

3. **Make topology changes emergent**
   Sparks and collapses should appear as:

   * new basins forming,
   * old basins disappearing,
   * basins merging or splitting, in response to the PDE, not as hard-coded basin surgery.

4. **Tie special events to mismatch, not geometry alone**
   Geometry (Hessian degeneracy, curvature) can identify where something interesting *might* happen, but RC’s sparks are driven by predictive insufficiency, not by curvature alone. Incorporating mismatch measures is essential in later stages.

5. **Separate “engineering” from “phenomenology”**
   The engineering problem—making a PDE stable and well-posed—is independent from the phenomenological problem—interpreting emergent patterns as organismal identities. This paper is about the former; only once it is solved is it safe to address the latter.

6. **Prefer functionals over ad-hoc terms**
   Whenever possible, derive the PDE as a gradient flow of a functional that encodes the desired trade-offs: basin depth vs curvature vs resource use. This provides a clear notion of what “stable” means.

## 9. Conclusion

Our initial attempts to implement Reflexive Coherence as a PDE system were dominated by two temptations: to force conceptual events (sparks, collapses) directly into the field, and to interpret any visually interesting pattern as an instance of ROM behavior. Both temptations led to instability and misinterpretation.

By systematically removing hard-coded event operators, enforcing basic numerical invariants, instrumenting the dynamics, and treating sparks and collapses as emergent phenomena, we arrived at a clearer understanding of what a “stable RC PDE” must look like: not a specific equation, but a family of equations constrained by theory and by numerical discipline.

The next step, taken up in a separate paper, is to investigate what these stabilized systems actually *do*: how basins form, interact, and reorganize over time, and to what extent their behavior matches the phenomenology predicted by ROM and RC.
