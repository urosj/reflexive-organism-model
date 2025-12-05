# **Emergent Dynamics in RC-Inspired PDE Systems**

**Part II — Observations, Phenomenology, and Failure Modes**

## **Abstract**

We present a phenomenological study of coherence-based PDE dynamics developed during the construction of a reduced Reflexive Coherence (RC) simulator. Unlike Part I—which documented the engineering process involved in achieving numerical stability—this paper analyzes the *behaviors* that arise once stable dynamics are obtained. Using a $128\times 128$) grid with $\Delta x = 0.1$ and timesteps ranging from $\Delta t = 10^{-5}$ to $10^{-3}$, we explore regimes characterized by diffusion, curvature-driven structure formation, nonlinear reserve coupling (“RC mode”), and optional perturbations (“sparks”). We observe: (i) early-time symmetry regeneration and rapid smoothing, (ii) mid-time filament and web formation for sufficiently large curvature parameter $\kappa$, (iii) universal coarsening toward one or few dominant basins, and (iv) the failure of naïve spark mechanisms to generate persistent new identities. Throughout, we annotate when, how, and why the system departs from RC intuition, turning the experiments into a methodological guide for interpreting coherence-like PDE dynamics.

## **1. Introduction**

Once a numerically stable RC-like PDE has been constructed (Part I), the next question is:
**What does such a system actually *do*?**

The present paper documents a series of empirical observations obtained from runs with varying parameter combinations, potentials, and initial conditions:

* many-basin initializations ($\texttt{basins}=100$),
* few-basin initializations ($\texttt{basins}=4$),
* double-well vs plateau potentials,
* pure PDE mode vs RC mode,
* sparks enabled vs sparks disabled.

These experiments are not intended to validate RC theory directly; instead, they serve as a phenomenological atlas of what one should expect when evolving coherence-like PDEs on a finite grid. This is indispensable groundwork for future attempts to align PDE behavior with the formal structure of RC.

## **2. Experimental Conditions**

We summarize the two main experiment families, with parameter sets explicitly listed.

### **2.1 High-resolution, small time step**

Used mainly for long, smooth evolutions:

* Grid: $512\times 512$
* Spacing: $\Delta x = 0.1$
* Timestep: $\Delta t = 10^{-5}$
* Steps: 500–2000
* Parameters: $\lambda = 0.8, \xi=0.1, \zeta = 0.05, \kappa = 0.05, \alpha = 1.0, \beta = 0.1.$
* Both *double-well* and *plateau* potentials used.

### **2.2 Coarse-resolution, large time step (RC mode)**

Used for web-like patterns and spark testing:

* Grid: $128\times 128$
* $\Delta x = 0.1$
* Timestep: $\Delta t = 10^{-3}$
* Steps: 100–300
* Parameters: $\lambda = 1.0, \xi=0.8, \kappa = 0.05, \alpha = 0.5, \beta = 0.8,  \gamma_{\text{curv}} = 0.3.$
* Reserve coupling enabled (`rc_mode = True`)
* Sparks tested with naïve Hessian-based criteria.

## **3. Early-Time Behavior Across All Regimes**

Across all experiments, regardless of potential or RC coupling, the first (10$–$30) iterations are dominated by **smoothing and reconstruction of symmetry**.

### **3.1 Diffusion dominates curvature and nonlinear terms initially**

With $\Delta t = 10^{-5}$, the diffusive term $\lambda \Delta C$ is the largest driver for several steps. Random initial basins (even 100 of them) rapidly lose their fine detail:

* small bumps spread and flatten,
* nearby bumps merge,
* noise decays quickly,
* the domain regains approximate rotational symmetry.

Even in RC mode with $\xi = 0.8$, the first few frames are visually indistinguishable from pure diffusion.

### **3.2 Double-well and plateau potentials behave nearly identically at early times**

Despite conceptual differences, early evolution is controlled by *local curvature* of $V(C)$ around the initial values of $C$. If the initial basins sit well inside the plateau region (for the plateau potential) or near the local minima (for the double-well), both potentials exert little local force:

* no sharp movement toward well-defined minima,
* almost linearized behavior,
* strong similarity between the two potentials for the first ~20–40 iterations.

This was explicitly observed in side-by-side runs with identical seeds but different potentials.

## **4. Mid-Time: Filament Formation and RC Signatures**

After the smoothing phase, the influence of curvature and nonlinear terms becomes visible.

### **4.1 Curvature-driven filament and web patterns**

For parameter sets with:

* $\kappa = 0.05$ or higher,
* $\gamma_{\text{curv}} = 0.3$,
* $\Delta t = 10^{-3}$,

we observe the emergence of:

* branching filaments,
* cross-shaped structures,
* web-like skeletons.

These resemble skeletonization instabilities seen in higher-order PDEs (e.g. Swift–Hohenberg-type behavior).

Characteristic timeline:

* **t ≈ 10–20 steps:** localized gradients begin to sharpen.
* **t ≈ 20–50:** filaments appear between basins.
* **t ≈ 50–100:** a network forms, connecting or partitioning coherence regions.

This is the only regime in which the PDE exhibits structure reminiscent of the “cosmic web” intuition discussed in earlier RC conceptual work.

### **4.2 RC coupling changes how filaments behave**

With `rc_mode = True` and $\xi = 0.8$, filaments become:

* thicker,
* longer-lived,
* more resistant to diffusion.

The reserve-like feedback amplifies coherent structures by rewarding regions that already have strong flux-like gradients.

Interestingly, the filaments **do not** preserve multiple identities; instead, they funnel coherence toward the dominant basin more efficiently, accelerating coarsening (§5.1).

## **5. Late-Time Regime: Coarsening, Spark-Induced Fragmentation, and Basin Motion**

Unlike classical reaction–diffusion systems that simply coarsen to a single attractor, the RC-inspired PDEs exhibit a **cyclic pattern** in late time:

1. **Coarsening toward a dominant basin**,
2. **Spark-induced fragmentation**, and
3. **Subsequent migration and recombination of basins**.

This cycle repeats across a wide range of parameters.

### **5.1 Coarsening still occurs — but not permanently**

As in standard gradient-flow systems, early and mid-time dynamics lead to:

* the emergence of one particularly strong basin,
* often located near a high-symmetry region of the domain.

This aligns with classical expectations: diffusion and nonlinear focusing naturally amplify the largest coherent structure.

However, unlike pure diffusive systems, **this dominant basin does not remain stable indefinitely**.

### **5.2 Sparks induce genuine fragmentation**

When spark events occur in the vicinity of an existing basin, they do not merely create noise or small perturbations. Instead:

* A spark in a weaker basin will often split that basin into **two or more local maxima**.
* If a **denser basin exists elsewhere**, the injected coherence frequently **flows into that denser basin**, effectively *feeding* the stronger identity but *altering* the shape and location of the weaker one.
* The result is a **repopulation of basins**:
  multiple distinct maxima re-emerge even after long coarsening phases.

Thus, sparks act as **dynamic fission events**, continuously reintroducing multiplicity into the system.

### **5.3 Basins move across the plane**

One of the most distinctive late-time phenomena is that **basins are not static**:

* peaks drift toward each other or along filamentary structures,
* a basin created by a spark may travel toward a stronger basin and merge with it,
* the dominant basin itself often **relocates** to a denser region after sparks reshape the landscape.

This migration demonstrates that the system supports a form of **effective curvature**:
coherence flows preferentially along directions influenced by the current mass distribution, not simply downhill in a fixed potential.

### **5.4 Dynamic multiplicity**

Together, these effects produce a regime where:

* the system oscillates between single-basin and multi-basin phases,
* sparks repeatedly break apart converged structures,
* basin identities move, merge, and re-form across the domain.

This late-time behavior is significantly richer than pure PDE coarsening and is the clearest indication that **nonlinear coupling and spark perturbations jointly produce a dynamic, field-dependent geometry** within the evolving coherence field.

## **6. Sparks: Fragmentation, Flow Redirection, and Field-Dependent Geometry**

The spark mechanism does not simply add coherence to the nearest basin.
Its interaction with the broader field structure produces three distinct phenomena:

### **6.1 Sparks split basins into multiple identities**

When a spark fires inside an existing basin:

* the basin often divides into **two or more local maxima**,
* the original location may weaken while a new maximum becomes dominant,
* the main basin may shift position, sometimes by several grid cells.

Thus, sparks act as **branching operators**, repeatedly reintroducing new identity candidates even after long periods of coarsening.

### **6.2 Sparked coherence flows toward denser basins**

A key empirical finding is that sparked coherence does **not** simply remain local:

* if a denser basin exists elsewhere in the field, the newly injected coherence tends to **drift toward that basin**,
* this flow follows **preferred channels** in the landscape, often along filaments or gradients shaped by curvature and RC coupling.

This demonstrates that the PDE already supports a form of **emergent curvature**:
the direction of flow is not symmetric but biased by the “shape” created by other identities.

### **6.3 Sparks generate motion in the identity landscape**

Because sparks reconfigure mass distributions and redirect coherence flow:

* basin positions change,
* some maxima vanish while others migrate,
* dominant basins may relocate entirely after several spark events.

The spark mechanism therefore induces a **mobile identity landscape**, rather than static or purely merging structures.

### **6.4 Sparks maintain multiplicity rather than eliminate it**

Contrary to initial expectations:

* sparks do **not** enforce monolithic identity,
* nor do they simply amplify the strongest basin.

Instead, sparks maintain a **dynamic population** of basins through:

* fragmentation,
* relocation, and
* flow-dependent recombination.

The result is a sustained interplay between **aggregation** (coarsening) and **differentiation** (spark-induced splitting).

### **6.5 Emergent Soft Curvature**

Although the geometry of the domain is fixed (Euclidean grid), the combination of:

* nonlinear RC coupling,
* spark-induced perturbations, and
* curvature-like operators (κ, γ_curv)

creates an **effective, field-dependent curvature**:

* coherence moves preferentially toward stronger basins,
* filaments guide flow,
* sparked mass follows nontrivial trajectories.

This is not yet the **reflexive, fully induced geometry** described in RC theory, where the metric itself is a functional of coherence.
However, it is a **precursor**: a soft curvature that shapes flow direction and identity motion in a manner reminiscent of RC geometric behavior.

## **7. Comparative Observations**

### **7.1 Double-well vs plateau in late time**

While early-time behavior was similar, the late-time differences were subtle:

* **Double-well:** dominant basin tends toward one of the wells; surrounding field collapses toward zero or the second well.
* **Plateau:** larger neutral region allows smoother transitions, but still ending in one local maximum.

In neither case did potential shape prevent coarsening.

### **7.2 RC mode only changes rate, not fate**

In all experiments:

* RC mode made structures sharper,
* speeded up dominance,
* produced more visually interesting mid-time patterns,
* **did not** change the final outcome: a single identity.

## **8. What These Observations Mean for RC Theory**

Although these PDEs were only inspired by RC—not faithful implementations—the observations teach several important constraints for future RC simulation efforts:

### **8.1 A pure PDE cannot maintain multiple persistent identities**

If RC theory demands a stable multiplicity of identity basins, then:

* the PDE must have *explicit anti-coarsening terms* or
* the model must be partly discrete (identity is nonlocal and semi-topological), or
* sparks/collapse must be emergent from a higher-level mechanism, not the PDE.

### **8.2 Geometry alone cannot generate RC sparks**

The Hessian-based spark rule demonstrated that local geometric degeneracy is not enough to create new identities:

* degeneracy is ubiquitous in high-dimensional fields,
* sparks must depend on *fit*, *mismatch*, or *incompatibility*, not just geometry.

### **8.3 RC collapse is not PDE collapse**

RC collapse is a *decision* of coherence flow; PDE collapse is slow geometric coarsening.
They are categorically different phenomena.

### **8.4 Curvature and RC coupling can create rich webs, but not identity separation**

Filaments are “free” in higher-order PDEs, but stable multi-identity structures are not.

## **9. Conclusion**

The RC-inspired PDE experiments reveal a late-time regime that is richer than simple coarsening and more structured than a static “one-basin” attractor.

Across parameter sweeps and initializations, we consistently observe a **cyclic pattern**:

1. **Smoothing and coarsening** create one (or a few) dominant basins.
2. **Sparks fracture** these basins, creating new local maxima and sometimes shifting the dominant peak to a denser location.
3. **Coherence flows along preferred paths** toward stronger basins, guided by an effective landscape shaped by curvature-like terms and RC coupling.
4. **Basins migrate and merge**, with some identities vanishing and others relocating, before the system partially coarsens again.

Rather than converging to a single static identity, the system remains in a **dynamical multi-basin regime**, with identities that move, split, and recombine over time. Sparks are not merely noise or a source term; they act as **recurrent branching events** that continually repopulate the identity landscape. The flow of sparked coherence into denser basins shows that the PDE realizes a form of **field-dependent “soft curvature”** on the plane: coherence does not diffuse isotropically but follows channels determined by the current configuration of basins and filaments.

At the same time, there are clear limitations relative to full RC:

* The geometry remains tied to a fixed Euclidean grid; curvature is expressed as an effective potential landscape in $(x,y)$, not as a fully reflexive, coherence-induced metric.
* Sparks and collapses are implemented as external rules, not derived from intrinsic geometric bifurcations.
* Identity basins are defined as maxima of $C(x,y)$, rather than attractors in an internal manifold whose shape is itself determined by coherence.

Taken together, these observations position the current PDE implementation as a **useful phenomenological proxy**: it already exhibits fragmentation, mobile identities, and flow guided by an emergent landscape, but it stops short of the full inverted-geometry picture of RC. The experiments therefore serve a dual purpose:

* they provide a **concrete playground** where one can watch coherence fields form, move, and interact;
* and they delineate the gap that remains to be closed by future models that incorporate *explicit* coherence-induced geometry, nonlocal functionals, or discrete reflexive structures.

In that sense, the present work should be read as a **bridge**: from standard PDE intuition toward the reflexive, geometry-creating dynamics required by RC, with the behavior of sparks, basins, and their motion providing the empirical constraints that any next-generation RC engine will have to respect.
