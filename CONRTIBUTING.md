# CONTRIBUTING.md  

**Thank you for helping improve the Reflexive Organism Model!**  
Below is a step‑by‑step guide on how to submit **paper corrections**, **experimental proposals**, **model extensions**, and **code examples**. Follow the workflow so that every contribution can be reviewed quickly and merged cleanly.

---  

## 1️⃣ General Workflow (All Contribution Types)

| Step | Action |
|------|--------|
| **Fork** the repository → **Clone** locally. |
| Create a **new branch** with a clear name, e.g. `fix‑typo‑section2`, `exp‑ant‑field‑dynamics`, `add‑assembly‑operator`. |
| Write your changes (markdown, LaTeX, code, data). |
| Commit with a concise message following the pattern `<type>(scope): <subject>` (see examples below). |
| Push to your fork and open a **Pull Request** (PR) against `main`. |
| Fill in the PR template. |
| Respond to reviewer comments; once approved, the maintainer will merge. |

---  

## 2️⃣ Submitting Paper Corrections  

### What belongs here?

- Typos, broken LaTeX commands, missing references.  
- Mis‑labelled equations or unclear phrasing (e.g., the upward coupling term in Eq. (1) could be clarified【Citation 1】).  
- Minor conceptual inaccuracies that do not require a new section.

### How to format

1. **Open an Issue** titled `Correction: <short description>`.  
2. In the issue body, quote the exact passage and propose the revised text.  
3. When you’re ready, create a PR that updates the LaTeX/Markdown source file.  

#### Example commit message  

```
fix(paper): correct upward‑feedback term description in section XYZ
```

---  

## 3️⃣ Proposing Experiments & Empirical Validation  

The model is deliberately **experiment‑oriented**. Use this section to suggest concrete laboratory or simulation studies.

### Suggested experiment template  
| Field | Content |
|-------|---------|
| **Title** | One‑sentence description (e.g., “Measure information‑compression amplification in a mixed‑species ant colony”). |
| **Motivation** | Which part of the theory does it test? (e.g., the dynamic reward‑mixing λ‑mechanism). |
| **Hypothesis** | Quantitative prediction (e.g., upward feedback term should increase field amplitude by ≥ 15 % after each aggregation step). |
| **Method** | Experimental setup, sensors, data acquisition, or simulation details (include code repository if applicable). |
| **Metrics** | AI(t), EI(t), Δfield, minority‑impact index, etc. |
| **Expected outcome** | How results would support or falsify the model. |
| **References** | Cite relevant sections of the manuscript (e.g., §Operators: upward coupling). |

Create a new issue with the label `experiment proposal` and attach the filled template. If you have simulation code ready, open a PR that adds it to the `experiments/` folder (see Section 5).

---  

## 4️⃣ Suggesting Model Extensions  

Extensions may involve:

- **New operators** (e.g., a lifting operator for inverse mapping of coarse‑grained fields【Citation 3】).  
- **Additional hierarchical levels** or alternative embedding strategies.  
- **Alternative reward‑mixing schemes** beyond λ‑mixing.  
- **Coupling to external physics** (e.g., adding a gravitational term as an emergent field).

### Extension PR checklist  

1. Add a short description in `extensions/README.md`.  
2. Include any new mathematical definitions with LaTeX, and reference the original sections they modify.  
3. Provide a **toy implementation** or pseudo‑code (see Section 5).  
4. Write a unit test that demonstrates the new component behaves as intended.  

#### Example commit header  

```
feat(extension): introduce stochastic lift operator L̂ for parent→child reconstruction
```

---  

## 5️⃣ Adding Code Examples  

The repository contains three main code directories:

| Directory | Purpose |
|-----------|---------|
| `src/` | Core library implementing the reflexive cycle (encoding ↔ field, upward/downward operators). |
| `examples/` | Minimal, runnable notebooks/scripts that illustrate a single concept. |
| `experiments/` | Larger simulation pipelines for proposed experiments. |

### What to contribute

- **Illustrative snippets** of the coarse‑graining operator \(\mathcal{Q}_\ell\) (e.g., Fourier low‑pass filter) as shown in the “fields as projections” discussion【Citation 2】.  
- **Full notebooks** reproducing a figure from the paper (e.g., upward coupling dynamics, minority‑effect amplification).  
- **Utility functions** for generating synthetic assembly pathways \(\mathcal{P}(t)\) and computing the Assembly Index AI(t).  

### Coding style

- Follow PEP 8 (Python) or the project’s ESLint config (JavaScript/TypeScript).  
- Use type hints (`def foo(x: np.ndarray) -> np.ndarray:`).  
- Document every public function with a docstring that includes a short description, parameters, returns, and an example.  

#### Example PR structure

```
examples/
├── 01_upward_coupling.ipynb          # visualises Eq. (1) from Citation 1
└── 02_minorities_butterfly_effect.py # reproduces minority‑triggered shift
src/
└── reflexive/
    ├── operators.py                  # add `lift_operator` implementation
    └── utils.py                      # helper for hierarchical embedding
tests/
└── test_lift_operator.py             # unit test confirming invertibility properties
```

---  

## 6️⃣ Issue & PR Templates (auto‑generated)

### Issue template

```markdown
**Type of contribution**
- [ ] Paper correction
- [ ] Experiment proposal
- [ ] Model extension
- [ ] Code example / bug

**Short description**

**Detailed description**
(Provide context, relevant equations, and any references to the manuscript.)

**Suggested solution / implementation (optional)**
```

### Pull Request template

```markdown
## Summary
A brief one‑sentence overview of what this PR does.

## Related Issue(s)
Closes #<issue-number>

## Changes Made
- List of modified files / added modules.
- Highlight new equations, operators, or experiments.

## How to Test
1. `pytest -q` (or appropriate command)  
2. Run the notebook/example (`jupyter nbconvert --execute ...`)  

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] New references cited correctly
```

---  

## 7️⃣ Community & Support  

* **Discord/Slack** – real‑time help, brainstorming sessions.  
* **Mailing list** – for longer discussions or announcements.  
* **Code of Conduct** – see `CODE_OF_CONDUCT.md`; we expect respectful and constructive interaction.

---

### Ready to contribute?  

1️⃣ Fork → 2️⃣ Branch → 3️⃣ Edit → 4️⃣ PR – make the Reflexive Organism Model both **theoretically solid** and **experimentally verified**.

*Have fun exploring!*
