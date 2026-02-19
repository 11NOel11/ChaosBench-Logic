# Future Work

Items deferred from v2.0.0 for future releases.

## v3 Candidates

### 5-Hop Reasoning Chains
The axiom graph's longest requires-chain is `Chaotic -> PosLyap -> Sensitive -> PointUnpredictable -> Bounded` (4 hops). 5-hop chains require additional axiom edges, e.g., adding a `Recurrent` predicate: `Chaotic -> Mixing -> Ergodic -> Recurrent -> Bounded` (5 hops). This needs careful mathematical validation before implementation.

### Heldout Template Population
The `heldout_templates` split is currently empty (HELDOUT_TEMPLATE_HASHES = empty set). Future work should identify a set of template hashes to hold out for compositional generalization testing.

### Extended Systems Expansion
Currently only 15 underrepresented systems are targeted for extended_systems questions (45 total). Future work should expand to all 30 core systems and potentially dysts systems.

### 15-Predicate Atomic Coverage
The v2 dataset was generated before the 4 new predicates (Dissipative, Bounded, Mixing, Ergodic) were added to PREDICATES. A full regeneration would produce atomic questions covering all 15 predicates rather than the original 11.

### v2 Model Evaluation Campaign
Published results are from v1 (621 questions). A full evaluation campaign on the v2 dataset (40,886 questions) across GPT-4, Claude, Gemini, and open-source models is needed.

### Class Balance Optimization
Two small families (extended_systems: 45, cross_indicator: 67) have class balance outside the [30%, 70%] range. Future regeneration should target better balance or apply the small-sample exemption more formally.

### Open-Book Evaluation (Out of Scope for v2)

The current benchmark is **closed-book only**: models are given natural language questions and must answer from training-time knowledge, without access to system equations, numerical solvers, or reference tables. Open-book evaluation (providing equations or computed time-series) is deliberately deferred: it would conflate the model's ability to *use* symbolic tools with its ability to *reason* about chaotic systems. Open-book settings are a natural future extension but are not planned for v2 or v3.

## Longer-Term

- **Multilingual support**: Translate questions beyond English
- **Open-ended evaluation**: Beyond binary TRUE/FALSE to numerical and explanatory answers
- **Visual reasoning**: Phase portraits and time series plots as inputs
- **Continuous parameter exploration**: Bifurcation diagram questions
