---
title: DemARK
description: Twenty-one worked demonstrations of heterogeneous-agent consumption and saving models, built on HARK
site:
  # The rails are position:fixed and overlay the margins, which landing blocks run underneath
  # rather than around. Only the outline is dropped here: hide_toc would take the contents out of
  # the page, and on a phone that element is the drawer. theme.css hides its rail instead.
  hide_outline: true
  # The hero below is the title. Without this the page shows the frontmatter title block first, so
  # a reader meets "DemARK" twice before reading anything.
  hide_title_block: true
---

+++ { "kind": "centered", "class": "ark-hero col-screen" }

# Models you can open and change

Twenty-one notebooks that build consumption and saving models one assumption at a time, from a
two-period problem to a life cycle with uninsurable risk.

{button}`Start with the Fisher two-period problem <./fishertwoperiod/>`
{button}`See all twenty-one <#start-here>`

+++ { "class": "ark-section col-body-outset" }

<!-- The label on each section below is a deep-link target, so a reader can send someone straight to
     the shocks group. The hero's second button uses this one; the rest are reachable by hand. -->

(start-here)=

## Start here

These four make a reading path for anyone new. The theory comes first, in a chapter of *A Gentle
Introduction to Intertemporal Choice*, then a two-period problem solved by hand, then that same
model in code, then income risk.

::::{grid} 1 1 2 2
:class: ark-steps

:::{card}
:link: https://intertemporal-choice.github.io/content/consumption/keynesfriedmanmodigliani/
**Keynes, Friedman, Modigliani**
^^^
Three consumption theories the rest of the library descends from, set against US data.
:::

:::{card}
:link: ./fishertwoperiod/
**The Fisher two-period problem**
^^^
Consumption over two periods, solved by hand, so you can see what the solver later does for you.
:::

:::{card}
:link: ./gentle-intro-to-hark-perfforesightcrra/
**Perfect foresight in HARK**
^^^
Your first model in code, still deterministic. An agent, a solver, a consumption function.
:::

:::{card}
:link: ./gentle-intro-to-hark-buffer-stock-model/
**Buffer stock saving**
^^^
Income risk enters and the consumption function stops being a straight line.
:::

::::

+++ { "class": "ark-section col-body-outset" }

(life-cycle)=

## Life cycle and the data

Income and spending over a working life, set beside what households are measured doing.

<!-- Three columns from the medium breakpoint up, not two. A group of three in a two-column grid
     leaves the third card alone on a row, which the six-card group below never does. -->
::::{grid} 1 1 3 3

:::{card}
:link: ./lifecyclemodeltheoryvsdata/
**The life cycle model against the data**
^^^
Simulated profiles set beside what households actually do.
:::

:::{card}
:link: ./lc-model-expected-vs-realized-income-growth/
**Expected against realized growth**
^^^
Transitory and permanent shocks pull expected and actual income apart.
:::

:::{card}
:link: ./perfforesightcrra-savingrate/
**The saving rate under CRRA**
^^^
What perfect foresight implies for how much is put aside.
:::

::::

+++ { "class": "ark-section col-body-outset" }

(shocks)=

## Shocks, credit and constraints

Income a household cannot insure, and limits on what it can borrow against.

::::{grid} 1 1 2 3

:::{card}
:link: ./mpc-out-of-credit-vs-mpc-out-of-income/
**MPC out of credit against income**
^^^
Loosening a credit limit and handing over cash are not the same stimulus.
:::

:::{card}
:link: ./changeliqconstr/
**Tightening a liquidity constraint**
^^^
What happens to the consumption function when borrowing gets harder.
:::

:::{card}
:link: ./incexpectationexample/
**The persistent shock model**
^^^
What a household expects its income to be, when shocks do not wash out.
:::

:::{card}
:link: ./tractablebufferstock-interactive/
**The tractable buffer stock model**
^^^
Labor income risk in a model simple enough to move by hand, parameters exposed.
:::

:::{card}
:link: ./nondurables-during-great-recession/
**Nondurables in the Great Recession**
^^^
Whether the model accounts for the spending drop of 2008.
:::

:::{card}
:link: ./chinese-growth/
**China's saving rate**
^^^
Whether precautionary motives explain saving through a period of fast growth.
:::

::::

+++ { "class": "ark-section col-body-outset" }

(aggregates)=

## Aggregates, prices and data

Many households added up, prices coming back out, and the aggregate series beside them.

::::{grid} 1 1 3 3

:::{card}
:link: ./diamondolg/
**The Diamond OLG model**
^^^
Overlapping generations, capital accumulation and the golden rule.
:::

:::{card}
:link: ./lucas-asset-pricing-model/
**The Lucas asset-pricing model**
^^^
Prices that clear a market of identical agents holding a risky tree.
:::

:::{card}
:link: ./durables-vs-nondurables-at-low-and-high-frequencie/
**Durables against nondurables**
^^^
Growth rates compared quarterly and over ten years, where the two diverge.
:::

::::

+++ { "class": "ark-section col-body-outset" }

(calibration)=

## Calibration and estimation

Which parameter values the evidence pins down.

::::{grid} 1 1 3 3

:::{card}
:link: ./micro-and-macro-implications-of-very-impatient-hhs/
**Impatient households, micro and macro**
^^^
What one cstwMPC parameter does to a household, and to the wealth distribution.
:::

:::{card}
:link: ./alternative-combos-of-parameter-values/
**Alternative parameters in cstwMPC**
^^^
The same model under combinations of parameter values that each fit the data.
:::

:::{card}
:link: ./structural-estimates-from-empirical-mpcs-fagereng-/
**Structural estimates from empirical MPCs**
^^^
Turning a reduced-form MPC estimate into structural parameters.
:::

::::

+++ { "class": "ark-section col-body-outset" }

(method)=

## Method and speed

For readers who came for the algorithms.

::::{grid} 1 1 3 3

:::{card}
:link: ./dcegm-upper-envelope/
**The DCEGM upper envelope**
^^^
Solving a discrete choice by endogenous gridpoints, with the kinks that leaves.
:::

:::{card}
:link: ./harmenberg-aggregation/
**Harmenberg aggregation**
^^^
A change of measure that needs a hundredth as many agents for the same precision.
:::

:::{card}
:link: ./perfforesightcrra-approximation/
**Approximating CRRA**
^^^
How close the approximate consumption function gets before it parts from the exact one.
:::

::::

+++ { "class": "ark-section col-body-outset" }

## Running them

A **Launch kernel** control sits on the right above the first cell of each notebook. It connects the
page to a session on Binder, and from then on you can run and edit the cells in place. Binder builds
that session on request, which takes a few minutes the first time. The rocket icon in the row of
links above the title opens the notebook somewhere else instead, in an external interface away from
this site.

To work offline, or to keep your changes, clone the repository and run JupyterLab locally. The
[README](https://github.com/econ-ark/DemARK#install-and-run-locally) has the steps for uv, conda and
Docker, and the environment each notebook runs in.
