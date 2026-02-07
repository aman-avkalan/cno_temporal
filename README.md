# CNO Temporal Training Pipeline (Lid-Driven Cavity)

## Overview

This project implements a **CNO (Convolutional Neural Operator)** model for the **2D Lid-Driven Cavity Navier–Stokes problem**, trained using **PyTorch** and orchestrated using **Temporal workflows**.

The goal is to demonstrate how **GPU-based scientific machine learning training** can be wrapped inside a **fault-tolerant, resumable, and scalable workflow system**, suitable for production or large-scale experimentation.

This repository is designed to be run on a **remote GPU machine via VS Code Remote SSH**.

---

## Problem Description

The Lid-Driven Cavity (LDC) problem is a classical benchmark in computational fluid dynamics (CFD).

- A square cavity filled with fluid
- Top wall moves horizontally (lid)
- Other walls are stationary
- Governing equations: incompressible Navier–Stokes

### Learning Task

- **Input (X)**: 3-channel spatial fields (problem configuration)
- **Output (Y)**: 4-channel solution fields:
  - u-velocity
  - v-velocity
  - pressure
  - auxiliary / derived channel

The model learns a **mapping from input fields to full flow solutions**.

---

## Why Temporal?

Training deep learning models is:
- Long-running
- Failure-prone (OOM, SSH disconnects, node restarts)
- Expensive (GPU time)

Temporal provides:
- Deterministic workflows
- Automatic retries
- Clear separation of orchestration vs execution
- Production-grade reliability

In this project:
- **Workflow** handles orchestration
- **Activity** runs GPU training
- **Worker** safely executes blocking PyTorch code

---

## Project Structure

```text
cno_temporal/
├── activities.py      # GPU training logic (PyTorch, CNO model)
├── workflows.py       # Temporal workflow definition
├── worker.py          # Temporal worker (executes activities)
├── run_workflow.py    # Client to start the workflow
├── requirements.txt
└── outputs/
    └── ldc_groundtruth_vs_prediction.png
```
## Model Architecture

The model is a simplified **CNO-style neural operator**, consisting of:

- Lift layer (input projection)
- Residual convolutional blocks
- Projection layer (output fields)

Key characteristics:

- Fully convolutional
- Resolution-independent
- Suitable for operator learning

The architecture is intentionally minimal to focus on **workflow orchestration rather than model complexity**.

## How to Run

### 1. Start Temporal Server

`temporal server start-dev`

---

### 2. Start Worker (GPU node)

`python worker.py`

---

### 3. Run Workflow

`python run_workflow.py`
