# <span style="color:#8C1A11">SIM</span><span style="color:green">P</span><span style="color:blue">L</span><span style="color:#f7ce46">E</span>

SIMPLE stands for
<span style="color:#8C1A11">SIM</span>ulation-based 
<span style="color:green">P</span>olicy 
<span style="color:blue">L</span>earning and
<span style="color:#f7ce46">E</span>valuation

Our goal is to build a simulation platform for policy learning and evalutions, featuring

* Diverse Built-in Tasks

    + Provide a wide range of environments for data collection, imitation learning, and evaluation.

    + Cover tasks from low-level motor control to high-level decision making, enabling cross-domain generalization.

* Evaluation-Oriented Architecture

    + Modular design for quickly creating real-to-sim evaluation environments.
    + Support for diverse robotic embodiments (manipulation, locomotion, multi-agent, etc.).
    + Focus on evaluation first to ensure fairness and replicability before optimization.

* Exhaustive Benchmarking Metrics

    + Standardize evaluation protocols with clear, reproducible metrics.
    + Support comparisons across state-of-the-art methods.
    + Enable community-driven leaderboards for transparent progress tracking.
    + Emphasis policy performance alignment between Real and Sim.

<!-- ## Table of Contents -->

```{toctree}
:maxdepth: 2



tutorials/index.md
robo-nix.md
user-guides/index.md
core/index.md
tasks/index.md
workflows/index.md
troubleshooting.md
developer.md
license.md
