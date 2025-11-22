# Coherence in Reflexive Organism Model

Copyright © 2025 Uroš Jovanovič, CC BY-SA 4.0.

The document is a summary of an investigation about the effects of attractors from which a deeper understanding of the role of coherence emerged, leading to the updated equation of effective stress tensor $T^{\text{eff}}_{\mu\nu}$ and the organism-level field equation $G_{\mu\nu}$.

## Abstract

This paper introduces coherence as the key invariant in the Reflexive Organism Model (ROM), showing that identity is preserved not by structure, experience, or compatibility alone but by their conserved product. Representing this coherence as a scalar field and tensor on the organism’s internal manifold clarifies how seed-imposed curvature shapes eigenmodes and attractor basins, extending the geometric intuition already present in ROM. While reserve dynamics determine how the organism grows, the geometric formulation illuminates why the transformations remain coherent. The result is a compact refinement of ROM’s core insight that organisms maintain identity by preserving coherence while continually reorganizing their internal structure.

## Background

What meaning do the seeds and eigenmodes have for organisms in terms of information processing? What changes in terms of presentation of "programs", and especially about activity selection? The effects of reserves, accumulation and flow. These were the questions that I was exploring when an issue arose which made me think about structural changes. When is existing structure good enough and when do you know if you've hit a plateau, saturation? When is good to act and spend reserves to introduce new structure, and when it is better to keep the existing structure but instead change the method, or move with a slower pace of growth? I started formulating the solution through usual risk assessment, when I realized that the two topics are actually very much a mirror of one another and that I should approach "risks" from the perspective of seeds and eigenmodes; unless one thinks in terms of attractors, everything is "an average sum of everything else", we're operating in a state of undifferentiated flux, a blur of average activity with no stable patterns.


## Information processing

Let's start with examining the role of seeds from the perspective of information processing.

A **seed** is a *minimally sufficient template* that encodes a high-probability pattern the system can reliably instantiate across iterations. In information-theoretic terms, seeds represent **lossy compression of experience into generative kernels**. They distill past coherence into compact forms that reduce future computational load. For an organism or program, creating seeds means shifting from *relearning* to *recalling*. Instead of recalculating behavior from first principles every cycle, it reuses proven configurations. This is analogous to subroutine formation in programming: once a useful function (e.g., edge detection, foraging logic) emerges and proves robust across cycles, the system "packages" it into a callable module. The seed becomes that module, a self-contained unit of action-perception coherence.  Without such compression every decision requires full recomputation, memory remains flat, there are no hierarchy of abstraction forms and there’s no bias toward efficiency, no “preferred paths” through possibility space.

The seeds thus encode the organism's bias toward reusing familiar patterns. This bias is not a limitation. It's a computational necessity. It enables *bounded rationality*, where agents navigate complexity by leveraging high-utility priors.

While seeds are static templates, **eigenmodes** describe dynamic stability, a self-consistent oscillations in the reflexive loop that sustain themselves via feedback resonance $\mathcal{R}[\psi_i] = \lambda_i \psi_i$. Here, $\psi_i$ is a pattern of organization (perception-action cycle) that outputs *itself*, slightly amplified. When active, it reinforces its own conditions for existence.

In information processing terms **eigenmodes are attractors in cognitive phase space.** They define trajectories the system naturally "falls into" when certain seeds are activated. They allow predictive stability, reducing entropy by narrowing future states to those within the basin of attraction while supporting long-term integration. This persistence enables continuity of identity and purpose. Even as components change.

Without eigenmodes, no feedback loop sustains itself long enough to become a functional unit as all processes remain transient, no emergent function accumulates. This all means that information is processed in isolation, there's no cumulative learning, no layering.

Another way to reason about attractors and lack of them is through sparse/structured processing. In case of **sparse structure (no seeds/eigenmodes)** information spreads thinly across possibilities. No pattern dominates and computation is diffusive. Decisions lack depth or continuity. This is akin to a neural network without strong weights, everything activates weakly, nothing coheres. On other hand, in case of **dense, seeded dynamics** specific configurations rise above noise due to prior success (seeded) and mutual reinforcement (eigenmodes). These form functional nuclei around which further complexity organizes. Like stars forming in a gas cloud via gravitational collapse, information collapses into dense attractors. Such dynamics is essential for memory (seeds store what worked), prediction (eigenmodes generate expectations) and scalability (hierarchical nesting allows higher-order cognition).

Hence, whether in biology or software, the system that grows doesn't rely on statistical analysis but rather discover seeds from successful subprocesses, stabilize them into eigenmodes via feedback loops, and protect reserves to explore beyond current basins.

To state this rather boldly, the need to form seeds and eigenmodes means that **intelligence is not in what you do once. It’s in what you can re-express, refine, and resonate over time**. Compression isn’t optional. It’s how systems escape randomness and begin to *mean something*.

## Building structures

Now that we've reasoned about what structures an organism builds and the meaning of attractors in terms of information processing. The next step is to start thinking about when is it appropriate to add new structures. Let's consider a case when an organism hits a "plateau". The structures are maximally utilized, but can't grow the usage anymore. Should it accumulate new structures, or build new structures from reserves or should it try to get by on a slower pace?

Let's state the problem formally. In the Reflexive‑Organism Model (ROM) a “plateau” is simply a *deep attractor basin* that has been filled to capacity. Every eigenmode that can be sustained by the current seed is already active, and the reflexive flux $\mathcal J$ is saturated.

This happens when all low‑cost seeds are already present and no new seed can be added without spending reserves. All eigenmodes with $\lambda_i>1$ are already saturated. Any further activity must either deepen an existing basin (hard) or create a *new* one. Looking at assembly,  $d AI/dt\approx0$. Any increase would require a *phase transition* in the underlying graph $\mathcal G$. The plateau persists because the current reserve level is just enough to keep existing basins alive but not enough to fund a new basin. On top of it, when the system cannot predict the payoff of a new structure, it must rely on EI to learn after the fact.

The model thus asks three questions:

1. Do we have enough reserves ($\rho_{\text{reserve}}$) to pay the cost of creating a *new* basin?  
2. What is the expected gain in assembly index $AI$ (i.e. how many extra, non‑trivial construction steps will the new structure add)?  
3. How uncertain is that gain – can we predict it beforehand or must we learn it retrospectively?

Based on state of reserves, curvature of its internal seeds and the alignment of fast fields with memory, we can explore three distinct ways in which the organism can rearrange, extend or preserve its coherence.

**Mode A: Generative Expansion (creating new structure).** When reserves are comfortably above threshold, the organism can afford to push outward into unexplored regions of its assembly space. In this mode, the policy $\Pi$ engages its growth operator and actively *adds new nodes and edges* to the history graph $\mathcal{G}$. A new structural seed is created, and with it a fresh eigenmode begins to form. 

Expansion is costly. The system must spend a visible portion of its reserve density $\rho_{\text{reserve}}$ to actuate the transition and supply the justification flux $\mathcal{J}_{AI}$ needed to stabilise the new structure. However, if the newborn mode crosses the critical threshold $\lambda>1$, it begins to reinforce itself in subsequent cycles. The assembly index $AI(t)$ increases by at least one irreversible step, and the organism’s reflective manifold deepens.

This mode is favoured when the reserve buffer is comfortably positive and the expected net gain in assembly depth outweighs the energetic overhead. It corresponds to the organism “trying a new possibility”, betting that the new eigenmode will be viable enough to pay for its own creation.

**Mode B: Internal Reconfiguration (deepening existing seeds).** When reserves are present but not abundant, the organism tends not to spawn new seeds but instead turns inward, applying its reflexive operator to *compress and reorganize the structures it already holds*. Rather than expanding $\mathcal{G}$, it searches for a more efficient basis of eigenmodes within the existing graph.

In this mode, no new assembly steps are taken, so $AI(t)$ stays flat. But the reuse efficiency and alignment of existing seeds improves. The eigenvalues $\lambda_i$ associated with established modes can drift upward as the system finds lower-dissipation pathways through familiar regions of state space.

This mode has a moderate energetic cost, most of it internal rearrangement rather than material expenditure. It is favoured when reserves are uncertain or when the cost of generating new structure would exceed the likely benefit. The organism chooses to refine its own geometry rather than extend it.

**Mode C: Maintenance / Coherence Preservation.** When reserves are depleted or highly volatile, the organism naturally shifts into a conserving regime. It reduces the consumption of $\rho_{\text{reserve}}$, allows the justification flux $\mathcal{J}$ to settle into a minimal-energy attractor, and uses what little surplus remains to support experience-driven refinement rather than structural growth.

Here, $AI(t)$ remains unchanged, and no attempt is made to push eigenvalues sharply upward. However, because the organism is not destabilising itself with aggressive structural moves, its experience index $EI$ can continue to grow softly. Over time, this leads to small but meaningful adjustments in the policy $\Pi$, increasing its predictiveness and internal coherence.

This mode arises whenever the risk of collapse is non-negligible, for instance, when the predictive horizon is short or the external field is volatile enough that a failed exploratory move could force self-termination. In this state, the organism opts not to grow, but to stay coherent until conditions improve.

No matter the mode, how should we treat **uncertain upgrades**? The model’s Intent Principle says the policy $\Pi$ should maximise the expected future assembly index 

$$
\Pi^{*}= \arg\max_{\Pi}\; \mathbb{E}[AI(t+\Delta t)\mid AI(t)] .
$$

When the expectation cannot be evaluated *a priori*, the model falls back on experience feedback $EI$ while reflexive loop records every trial in the slow memory field $M(t)$. After a trial finishes, the system updates its estimate of $\mathbb{E}[AI]$ and adjusts $\Pi$ accordingly. A possible heuristics would thus consist of the simple three steps:

1. **Sandbox trial** allocates a *small* reserve packet to build a provisional seed (a “pilot” eigenmode).  
2. **Observeation** of the resulting $\lambda_{\text{trial}}$. If $\lambda>1$ and the basin deepens, store the trial’s graph fragment in $M(t)$.  
3. **Retrospective update** of the policy. Increase the probability of choosing similar construction steps next cycle.

Thus *uncertainty* is resolved by a **controlled‑risk exploration** that keeps reserve consumption bounded while still allowing the system to discover new basins. Controlled-risk here is tied to reflexivity conservation law $\partial_t\bigl(\mathcal J(t)\,\rho_{\text{compat}}(t)\bigr)=0 .$ Any increase in the growth term $G$ (new structures) must be compensated by either by a temporary dip in $\rho_{\text{reserve}}$ (spending reserves), or a reduction in another flux (e.g., lowering background loss $L$). If the organism cannot afford the reserve dip and ther are no reserves to cut losses, the only admissible move is to keep $G=0$, that is to stay in maintenance mode C.

We can now define the conditions which can push the organism into a "greedy" mode. This is when the reserves are over-spent, and not only start to temporarily deplete, but creating a negative reinforcing loop. Or when too many structures are accumulated, but without forming any reflexing cycles. In other words, it is a state where the organism starts to act against the conservation law, not just temporarily, but creating states that reinforce the degradations.

The reflexivity conservation law is key to selection of whatever heuristic of growth is chosen.

## Coherence as the core quantity

We've been using notion of coherence throughout descriptions of the model. Let's make a formal definition of what it means. We already have the reflexivity conservation principle (RCP):

$$  
\frac{d}{dt}\Big[ \mathcal J(t)\rho_{\text{compat}}(t) \Big] = 0  
$$

with

$$  
\mathcal J(t) = \big(\mathrm{AI}(t)\big)^{\alpha(t)} \big(\mathrm{EI}^\star(t)\big)^{\beta(t)}, \qquad \rho_{\text{compat}}(F_t,M_t) = \frac{I(F_t;M_t)}{\sqrt{H(F_t)H(M_t)}} e^{-D(\mathcal F_t|\mathcal F_{\text{noise}})}.  
$$

We now name the product explicitly:

$$  
C_{\text{sys}}(t) \equiv \mathcal J(t) \rho_{\text{compat}}(t) \big(\mathrm{AI}(t)\big)^{\alpha(t)} \big(\mathrm{EI}^\star(t)\big)^{\beta(t)} \rho_{\text{compat}}(F_t,M_t).
$$

**Definition (Coherence scalar):**  $C_{\text{sys}}(t)$ is the _coherence_ of the reflexive organism at time $t$. It measures how much _meaningful structure_ the system carries (through AI and EI*), weighted by how well its fast field and memory are aligned (through $\rho_{\text{compat}})$.

Let's take a step back and explore why the scalar defined as a product matters. The three quantities carry most of the internal geometry of an organism. They define *what* the organism is made of (its structural history), *how* it uses what it is made of (its reflexive loop), and *whether* its current activity is compatible with its own past (its self-consistency). But none of these alone tells you whether the organism is coherent.

The organism’s stability does not depend on any one of them, but on the *simultaneous presence* of sufficient structure, sufficient experience, and sufficient compatibility. This is why the model does not preserve AI, or EI, or $\rho_{compat}$ individually. It preserves the product.

Only the product

$$
C_{\text{sys}} = (AI)^{\alpha}(EI^\star)^{\beta}\rho_{\text{compat}}
$$

captures the amount of structural memory present, the degree to which the organism is currently using that memory and the match between memory and the ongoing flow of activity.

This combination is not arbitrary. It is the **minimum quantity that fully captures the organism’s ability to remain itself through time.** It is the organism’s *coherent identity density*, what remains invariant while the organism reorganizes its internal space.

The model does not care about “structure”, “experience”, or “compatibility” in isolation.
It cares about the **coherence of the loop** that binds them.

RCP becomes simply:

$$  
\frac{d}{dt} C_{\text{sys}}(t) = 0
$$

for any intact organism, that is, any system that has not collapsed or undergone fission.

It is a **structural invariant** of ROM, a quantity that any intact reflexive trajectory must preserve in order to remain coherent. It plays a role *analogous* to a conservation law in physics, but it is not asserted as a universal physical law. Rather, it is a **law-like requirement internal to the definition of a reflexive organism**.

RPC now states that the organism may rearrange its entire internal geometry (grow new structure, deepen experience, reconfigure compatibility) yet the **overall coherence budget** carried by the system must remain balanced. What changes across time is *how* this coherence is distributed, not the fact that the organism maintains it. *Collapse* corresponds to the limit $C_{\text{sys}} \to 0$, where the reflexive loop can no longer sustain itself.  *Division* corresponds to a redistribution $C_{\text{sys}} \to C_1 + C_2$ across newly formed child fields. And *growth* corresponds to a reallocation of coherence across modes while the invariant is maintained.

In this sense, the invariance of $C_{\text{sys}}$ serves as the organism’s identity condition inside ROM: *a reflexive system is one that preserves its coherence budget across internal transformations.*

This “law-like” invariant gives ROM the structural requirement that prevents the organism from dissolving into a collection of unrelated updates and heuristics.

We can now define a **coherence field** $C(x)$ of the organism (the spatially distributed density of reflexive coherence)

$$  
C(x) \equiv C(x^\mu) = \mathcal J(x) \rho_{\text{compat}}(x),  
$$

where $x^\mu$ are spacetime coordinates on the organism’s manifold (e.g. tissue, colony, forest), $\mathcal J(x)$ is now a local justification density, and $\rho_{\text{compat}}(x)$ is the local compatibility of fast and slow fields at that point.

### Coherence continuity equation and flux

We already have a local continuity law for compatibility

$$  
\partial_t \rho_{\text{compat}}(x,t) + \nabla\cdot \mathcal J(x,t) = s(x,t) - \eta(x,t),  
$$

where $s$ is a source (injected coherence) and $\eta$ dissipation. We now want a **continuity equation** for the coherence density $C(x,t)$. The natural form is:

$$  
\partial_t C(x,t) + \nabla\cdot \mathbf{J}_C(x,t) = 0,  
$$

where $\mathbf{J}_C(x,t)$ is the **coherence flux**. We can define $\mathbf{J}_C$ as coherence flowing along the existing compatibility flux $\mathcal J$

$$  
\mathbf{J}_C(x,t) \equiv C(x,t) \mathbf{v}_C(x,t), \qquad   \mathbf{v}_C \sim \frac{\mathcal J(x,t)}{\rho_{\text{compat}}(x,t) + \epsilon}.  
$$

$\epsilon$ is just a small regularizer to avoid division by zero. The idea begind $\mathbf{v}_C$ is that $\mathcal J$ tells you _where_ compatibility flux wants to flow and $\rho_{\text{compat}}$ rescales it, so $\mathbf{v}_C$ is an effective “velocity” for coherence density. 

We can state then that

$$  
\partial_t C + \nabla\cdot\big( C\mathbf{v}_C \big) = 0  
$$

is a local conservation law: *any local gain of coherence density is exactly balanced by influx from neighbors; any loss is balanced by outflux*. 

The RCP becomes the statement that coherence is neither created nor destroyed inside the organism. It is _transported_ across space and _reallocated_ between assembly, experience, and compatibility.

We can integrate this over any region $\Omega$ of the organism:

$$  
\frac{d}{dt}\int_{\Omega} CdV  
= -\int_{\partial\Omega} \mathbf{J}_C\cdot d\mathbf{S}.  
$$

Flux of coherence through the boundary changes the coherence content inside (like charge or mass in a fluid).

### Coherence tensor $K_{\mu\nu}$

In the paper, we already extended the stress–energy tensor to include memory and active stress

$$  
T^{\text{eff}}_{\mu\nu}  
= T_{\mu\nu} - \kappa\left( \nabla_\mu M \nabla_\nu M - \frac{1}{2} g_{\mu\nu} (\nabla M)^2 \right) - p_{\text{act}}(\mathcal E) h_{\mu\nu} - j_\mu j_\nu.
$$

The paper also interprets $\rho_{\text{compat}}$ as the true measure of individuality as coherent interaction across scales. Now define a **coherence tensor** $K_{\mu\nu}$ that encodes how coherence is stored and flows in spacetime:

A minimal definition is

$$  
K_{\mu\nu} \equiv \underbrace{\lambda_C C g_{\mu\nu}}_{\text{coherence density term}} + \underbrace{\xi_C \nabla_\mu C \nabla_\nu C}_{\text{coherence gradient term}} +   \underbrace{\zeta_C j_\mu j_\nu}_{\text{read-back flux term}},  
$$

where $C(x)$: coherence density field,  $\nabla_\mu C$ are gradients of coherence (where coherence increases/decreases), $j_\mu$ is the existing read-back currents, the microscopic carriers of $\mathcal J$ and $\lambda_C,\xi_C,\zeta_C$ are  phenomenological coupling constants.

In the equation the term $\lambda_C C g_{\mu\nu}$ acts like an isotropic “coherence pressure”, a background tendency for the field to curve geometry wherever coherence is high. The term $\xi_C \nabla_\mu C \nabla_\nu C$ encodes how sharp coherence gradients themselves contribute to local “tension” in the field. Sharp transitions in coherence cost something and bend spacetime. The term $\zeta_C j_\mu j_\nu$ is a direct contribution of reflexive currents. Where justification flux is strong and aligned, coherence forms structured “beams” in the geometry.

We can then fold this into the effective stress tensor as an additional contribution

$$
T^{\text{eff}}_{\mu\nu} = T_{\mu\nu} + A_{\mu\nu}[AI] + \Theta_{\mu\nu}[EI] + K_{\mu\nu}[C] + \kappa\left( \nabla_\mu M \nabla_\nu M + \tfrac{1}{2} g_{\mu\nu}(\nabla M)^2 \right) + j_\mu j_\nu,
$$

so curvature responds not only to structure and active stress, but also **explicitly to coherence density and its gradients**. In words: **geometry is shaped by where coherence lives and how sharply it changes.** Deep attractor basins correspond to regions where $C$ is high and $\nabla C$ is small (low dissipation, stable coherence).

The organism-level field equation, including the coherence tensor, is

$$
G_{\mu\nu} = 8\pi G \left[ T_{\mu\nu} + A_{\mu\nu}[AI] + \Theta_{\mu\nu}[EI] + K_{\mu\nu}[C] + \kappa\left( \nabla_\mu M \nabla_\nu M - \tfrac12 g_{\mu\nu}(\nabla M)^2 \right) + j_\mu j_\nu \right].
$$

In this form:

* $T_{\mu\nu}$ describes physical matter,
* $A_{\mu\nu}$ encodes assembly-induced curvature,
* $\Theta_{\mu\nu}$ encodes experience-driven active stress,
* $K_{\mu\nu}$ is the **coherence-geometry term**,
* the memory gradient term shapes the organism’s internal manifold.

This is the fully extended ROM field equation, where coherence is now a first-class geometric source alongside matter, structure, experience, and memory.

### Geometric interpretation: coherence as curvature of reflexive space

We've already described the organism’s internal space as a curved reflexive manifold shaped by seeds. Seeds are encoded as fixed parameters $(\theta,D,g,\mathcal B)$. The reflexive operator $\mathcal R$ admits eigenmodes $\psi_i$ on this manifold.

We can now interpret $C(x)$ geometrically as a **scalar field** measuring _how strongly the reflexive manifold is “occupied” by self-consistent structure at each point_. Deep attractor basins represent regions where $C$ is high and nearly constant, while collapses are located where $C \to 0$ or large destructive gradients $|\nabla C|$.

Combined with the coherence tensor $K_{\mu\nu}$, regions with high $C$ and small $\nabla C$ act like “valleys” in the reflexive landscape: currents tend to flow along them with minimal dissipation (deep eigenmodes). Regions with steep $\nabla C$ are domain walls, boundaries between different patterns of being, where coherence is reconfigured (phase transitions, differentiation, decisions).

Thus the **coherence field $C(x)$** is the scalar that tells you how much of the manifold’s curvature is actually _inhabited_ by reflexive structure at a given time. Geometry encodes the potential structure of the organism, the curvature that defines all admissible modes, attractor shapes, and compatibility pathways. Coherence is the realized structure, the portion of this latent geometry that is actually inhabited, stabilized, and maintained by reflexive feedback. Geometry is possibility while coherence is actuality.


---

## Appendix A: Lagrangian density and Noether symmetry for coherence

To treat coherence as a proper field, we can introduce a **Lagrangian density** $\mathcal L_C$

 $$  
\mathcal L_C = \frac{\kappa_C}{2}\nabla_\mu C\nabla^\mu C - 
V(C) + \gamma_C C\mathcal S(M,\mathcal F),  
$$

where 

- $\kappa_C>0$: stiffness of the coherence field (how costly spatial variation is).
- $V(C)$: a potential that encodes preferred coherence levels (e.g. a double-well for bistable attractors).
- $\mathcal S(M,\mathcal F)$: a source term coupling coherence to memory and fast-field structure, for example

$$ 
\mathcal S(M,\mathcal F) \sim \rho_{\text{compat}}(F,M)\frac{I(F;M)}{\sqrt{H(F)H(M)}} e^{-D(\mathcal F|\mathcal F_{\text{noise}})}.
$$

The gradient term $\frac{\kappa_C}{2}(\nabla C)^2$ penalizes rapid spatial changes in coherence, favoring smooth, extended coherent regions. The potential $V(C)$ can encode a preferred coherence level (e.g. $C=C_0)$, or multiple preferred attractors. The coupling term $\gamma_C C \mathcal S$ says: coherence is _created/stabilized_ wherever compatibility between memory and dynamics is high.

The Euler–Lagrange equation for $C$ is

$$  
\nabla_\mu\left(\frac{\partial \mathcal L_C}{\partial (\nabla_\mu C)}\right) - \frac{\partial \mathcal L_C}{\partial C} = 0, 
$$

which yields

$$  
\kappa_C \Box C + \frac{dV}{dC} = \gamma_C \mathcal S(M,\mathcal F).  
$$

The $\Box C = \nabla_\mu \nabla^\mu C$ is the d’Alembertian (wave operator) and the coherence evolves as a driven, damped nonlinear field, pushed by compatibility sources $\mathcal S$.

If $\mathcal L_C$ is invariant under global shifts $C \to C + \text{const}$ or under some internal symmetry of the reflexive manifold, Noether’s theorem guarantees a conserved current $J_C^\mu$ with

$$  
\nabla_\mu J_C^\mu = 0.  
$$

We can then _identify_ $J_C^0 = C$ (coherence density) and $J_C^i = \mathbf{J}_C$ (coherence flux), exactly reproducing the continuity equation:

$$  
\partial_t C + \nabla\cdot\mathbf{J}_C = 0.  
$$

In other words, RCP arises as a Noether symmetry of the coherence field theory. The symmetry states: _we can redistribute coherence between assembly, experience, and compatibility without changing the total coherence content_. The corresponding conserved current is exactly the coherence flux.


## Appendix B: A 1D Coherence-Field Toy Model

To illustrate how the coherence field $C(x,t)$ behaves in the simplest possible setting, consider a one-dimensional reflexive substrate $x \in [0,L]$ with a single seed $\psi(x)$ located near the center. We model the coherence dynamics using the coherence continuity equation

$$ 
\partial_t C(x,t) + \partial_x J_C(x,t) = 0,  
$$

with coherence flux  

$$  
J_C(x,t) = C(x,t) v_C(x,t), \qquad  v_C(x,t)\frac{\mathcal J(x,t)}{\rho_{\text{compat}}(x,t) + \varepsilon},  
$$  
where $\varepsilon$ buffers against vanishing compatibility.

To keep the example minimal, we assume:

- **constant structural depth** over space  $\mathrm{AI}(x) = A_0,$
- **local EI$^\star$** tied to the seed shape $\mathrm{EI}^\star(x) = E_0 \exp\left( -\frac{(x-x_0)^2}{2\sigma^2} \right),$    
- **compatibility** that grows where the field aligns to the seed $\rho_{\text{compat}}(x,t) R_0 + \lambda C(x,t).$

This is enough to illustrate the formation of a coherent attractor without introducing new dependencies. Under these simplifications, the coherence scalar density becomes

$$  
C(x,t) = \big(A_0\big)^{\alpha} \big(\mathrm{EI}^\star(x)\big)^{\beta}  
\big(R_0 + \lambda C(x,t)\big).  
$$

With the above definitions, the coherence flux is

$$  
J_C = C(x,t) \frac{ A_0^\alpha \left(\mathrm{EI}^\star(x)\right)^\beta  
}{ R_0 + \lambda C(x,t) + \varepsilon }.  
$$

Therefore,

$$  
\partial_t C(x,t) = - \partial_x \left[ C(x,t) \frac{ A_0^\alpha  
\left(\mathrm{EI}^\star(x)\right)^\beta }{ R_0 + \lambda C(x,t) + \varepsilon } \right].  
$$

This is a nonlinear conservation law with seed-driven drift. The _seed_ $\mathrm{EI}^\star(x)$ generates a spatially varying “pull” toward $x_0$. Linearizing around small $C$, the drift reduces to

$$ 
v_C(x) \approx \frac{A_0^\alpha\left(\mathrm{EI}^\star(x)\right)^\beta}{R_0+\varepsilon},  
$$

so coherence flows **toward the maximum of $\mathrm{EI}^\star(x)$**, which is at $x_0$. Thus coherence starts diffusing inwards and accumulating around the seed. As $C$ grows locally, the term $\lambda C$ in $\rho_{\text{compat}}(x,t) = R_0 + \lambda C(x,t)$ becomes significant. This increases compatibility, which in turn reduces the velocity $v_C$, effectively **slowing the flow and trapping coherence** around the seed region. The system reaches a steady state when $\partial_x J_C = 0, \ \Rightarrow\ J_C = \text{const},$ and since boundaries are reflective or no-flux, we get

$$ 
J_C=0 \qquad \Rightarrow \qquad C(x,t) \partial_x v_C(x,t) + v_C(x,t) \partial_x C(x,t)=0. 
$$

The physically meaningful solution is $\partial_x C(x) = 0$ where $v_C=0.$ The drift velocity $v_C$ vanishes when $A_0^\alpha \left(\mathrm{EI}^\star(x)\right)^\beta = 0 \ \Rightarrow \ x = x_0.$ But because $\rho_{\text{compat}}$ increases with $C$, the region where $v_C$ gets suppressed **broadens** around $x_0$.  This produces a **stable coherence bump**

$$  
C_{\star}(x) = C_0 \exp \left(-\frac{(x-x_0)^2} {2\sigma_{\mathrm{eff}}^2}\right),  
$$

where $\sigma_{\mathrm{eff}}$ is _larger_ than the seed width $\sigma$. This is the hallmark of an attractor, the coherence field becomes **wider and deeper** than the seed that generated it.

This simple example mirrors the higher-dimensional ROM phenomena which encapsulates attractor formation, stabilization of eigenmodes, coherence-driven flux redistribution, and self-deepening of structure.


## Appendix C: Destructive agents

Let's define a “destructive” organism as one where

$$
\lambda_{\text{ind}} \gg 0,\quad
\rho_{\text{compat}}^{(\text{sub, parent})} \ll 1.
$$

Its own loop is closed and strong (it remembers itself, acts consistently), however its behavior **reduces** the parent’s compatibility / coherence. In addition, $\lambda$-mixing for altruistic vs exploratory behavior 

$$
G_{\text{eff}}(t) = (1-\lambda(t))G_{\text{coop}} + \lambda(t)G_{\text{expl}}
$$

state the $\lambda$(t) for an agent is close to unity. Its gain is dominated by exploratory/selfish part $G_{\text{expl}}>1$, so it can drive its own eigenvalue above unity even if that hurts the global field.

So the destructive agent is mathematically an eigenmode with $\lambda_i>1$ within the sub-field,  but whose net contribution to parent’s $C(t)$ and $R$ is negative 

$$
\frac{d}{dt} C_{\text{parent}} < 0, \quad \frac{d}{dt} R_{\text{parent}} < 0
$$

Locally coherent but globally incoherent. How do colonies and ecosystems suppress destructive agents? Let's recall reserves at layer $L_k$

$$
\partial_t R_k = \mathcal D_k - \mathcal U_k - \eta_k
$$

with $\mathcal{D}$ downward inflow (resources, constraints) and $\mathcal U$ upward coherence export. For a sub-agent $j$ inside layer $k$, it receives some share of $\mathcal D_k$ and it contributes to $\mathcal U_k$ according to its compatibility and EI. If agent j is destructive 

$$
\text{(for j)}:\quad \lambda_{\text{ind},j}>0,\quad \Delta \mathcal U_k^{(j)} \ll \Delta \mathcal D_k^{(j)}
$$

i.e., it consumes more coherence than it returns. Accumulated effect becomes $\partial_t R_k < 0$. The hierarchy then has a strong incentive (via its own RCP) to reduce $\mathcal D_k^{(j)}$ (cut resources, access, influence), change $\lambda$-mixing to lower agent $j$’s effective gain, or isolate or remove $j$ (biological apoptosis, immune response, social policing, ecological starvation). In other words, at the colony/ecosystem scale each agent is an eigenmode in a larger operator. Modes that repeatedly cause $\frac{d}{dt} C_{\text{parent}}<0$ or $\partial_t R_{\text{parent}}<0$ are **pruned** or suppressed by the dynamics that keep parent-level RCP viable. 

## Appendix D: Mathematics of emotions

And now for something completely different. Let's consider coherence from a completely different perspective and make a hypothesis that in ROM, coherence is the feeling or condition where all parts of a system, that being thoughts, actions, memory, or perception, are aligned and reinforcing each other. When something feels “right,” it’s because it fits one's internal structure. When something feels “wrong,” it’s because it destabilizes that structure. This feeling becomes the main steering force of the organism.

In this view, we can state the difference between *insight* and *greed* comes down to how reinforcement interacts with coherence. *Insight* is reinforcement that strengthens coherence. It feels calm, clear, stable, and internally aligned. *Greed* on the other hand is reinforcement that *breaks* coherence. It feels urgent, pressured, noisy, and destabilizing. Both are “amplifications,” but only one supports long-term stability.

So insight corresponds to a mode with $\lambda_i>1$ (self-reinforcing), but still compatible with the rest of the organism, keeping $C(t)$ preserved and reserves not drained. 

Greed on the other hand corresponds to a mode where $\lambda_i > 1$ for some eigenmode (strong reinforcement), but that mode drives down either $\rho_{compat}$ or EI* for the whole $\frac{d}{dt} C(t) < 0$ or $\frac{d}{dt} R(t) < 0$. If the reinforced mode causes net negative $\frac{dR}{dt}$ reserves flow while boosting its own gain, you get a locally amplifying but globally coherence-breaking pattern.

Similarly, love and fear turn out to be the same structural distinction. *Love* is the state of expansion resulting in coherence rising, openness, connection. While *fear* is the state of contraction, sensing coherence falling, protection, narrowing. Fear isn’t “bad.” It’s how the organism tries to stop its internal coherence from degrading further. This gives an elegant structural explanation of emotional life.

Formally, love equals coherence expansion regime, when $\frac{dC}{dt} \ge 0$ and EI* is high with feedback loops well-grounded in memory and prediction metrics (mutual information + transfer entropy in EI definition are high) and the reachable volume of coherent states $|\Omega_{\text{coh}}|$ grows or stays large. Here $\Omega(t)$ represents the accessible region of state space. Hence, in this mode, the following holds $\frac{d}{dt}|\Omega_{\text{coh}}(t)| \ge 0$ and the system can afford to stay open (engage with the environment, absorb perturbations, try new configurations, widen its working set, allow exploratory dynamics, and tolerate uncertainty).

Fear emerges during degradation and contraction. It is the opposite regime, where $\frac{dC}{dt} < 0$ and to prevent full collapse, the system contracts. It either effectively decreases conductance $D$ (flows restricted), narrows the attractor basins (accessible states shrink), or tries to minimize further damage to $C$. So fear is the organism entering a geometry-protection mode when it reduces the size of $\Omega$ it explores to prevent $C(t) \to 0$. This fits entropy pump too. When $\Phi_{\text{EI}}$ drops, system can no longer keep local entropy low, so it reduces openness ($D$, exploration) to stop decoherence.

We've already shown that decision-making comes from the seed, not statistics. One can even observe that that choosing something important based on formulas or risk calculations often feels empty. Why? Because coherent decisions must align with the deep attractor basins (the “seed”), the long-lived internal patterns that shape your identity and behavior. Only when a decision resonates with these deep structures does it feel meaningful.

Decisions that “feel right” correspond to actions that project strongly onto the seed-aligned modes. Let's define

$$
A \mapsto \mathcal J_A(t) \quad\text{(justification flux under action A)}.
$$

Then define a **seed alignment functional**

$$
\mathcal A(A) = \left\langle \psi_{\text{seed}}, \mathcal J_A \right\rangle
$$

When $\mathcal A(A)$ is large and positive action $A$ lies along the seed’s preferred direction in the $(\mathcal J,\rho_{\text{compat}})$ manifold, and it requires minimal structural “torque” to realize, and it preserves RCP with low dissipation. 

A purely statistical / external optimization that yields a $\mathcal J_A$ almost orthogonal to $\psi_{\text{seed}}$ will technically increase some external utility, but produce low $\mathcal A(A)$, so it feels empty or wrong (low internal coherence).

Based on the model, the structures, and flows, how does one feel or detect which side one is acting from?

When the acting comes from the seed, the feeling is quiet, clean, stable. Not hyped, not rushed, not urgent. It feels like a click, not a push. The action deepens clarity rathern than flooding the mind. It organically recall past patterns, long-term values, structural intentions. It feels like the decision is consistent with years, not minutes. And the decision would make sense even if there's no "external reward". Instead, you sense increase of internal reserves. And seed-aligned decisions don’t decay with time while greed evaporates when the acceleration fades.

When the acting comes from greed, it feels urgent, pressured and accelerating. It always has a push, rush, also an addictive edge. It feels like contraction. Greed is decontextualized action, there is no reference to slow memory. The actions often bring overthinking, excuses, justification. Insight does not require a story, greed always does and require external payoff to feel valid.

This hypothesis shows that there are only two global modes in a reflexive organism: coherence reinforcement or coherence breakdown (runaway collapse). Everything else (love, intuition, greed, fear, insight, burnout) is an *expression* of which side of this bifurcation the system is currently on.

---

## Reflection

The paper finished with what actually started it. First there was an emotion triggered by breaking of structure. And it was observation of emotions that led towards how one make decisions. The core, the mathematics of coherence came only at the end. Looking back, this is how the seed is created. On one hand, the area of content needs to be exposed first. You move on the curvatures that lead to something from many different angle, but they all are converging towards the same point. And then compression happens and the seed crystalizes. The core of the idea that was only a hunch in the beginning.

---

## Bibliography

- **Jovanovic, U.** (2025). *Reflexive Organism Model*.
- **Jovanovic, U.** (2025). *Seeds of life*
