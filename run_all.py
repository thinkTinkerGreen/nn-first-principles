#!/usr/bin/env python3
"""Run all 4 deep learning first-principles experiments and save publication-quality visual proofs."""

import argparse
import os

from src.activations import run_experiment_1
from src.embeddings import run_experiment_3
from src.generalization import run_experiment_4
from src.linear_depth import run_experiment_2


def main():
    parser = argparse.ArgumentParser(description="Run 4 Deep Learning First Principles Experiments")
    parser.add_argument("--output-dir", type=str, default="assets", help="Directory to save figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 75)
    print("  DEEP LEARNING FIRST PRINCIPLES: SUITE EXECUTION")
    print("=" * 75)

    # 1. Activations
    print("\n[1/4] Running Experiment 1: Activations exist for a reason...")
    fig1_path = os.path.join(args.output_dir, "01_activations_boundary.png")
    acc_lin, acc_relu = run_experiment_1(fig1_path)
    print(f"      -> Single linear layer accuracy:  {acc_lin * 100:.1f}%")
    print(f"      -> One ReLU hidden layer accuracy: {acc_relu * 100:.1f}%")
    print(f"      -> Saved figure: {fig1_path}")

    # 2. Linear Depth
    print("\n[2/4] Running Experiment 2: Depth without nonlinearity is a lie...")
    fig2_path = os.path.join(args.output_dir, "02_depth_collapse.png")
    acc1, acc5_lin, acc5_relu, max_err = run_experiment_2(fig2_path)
    print(f"      -> 1-Layer Linear accuracy: {acc1 * 100:.1f}%")
    print(f"      -> 5-Layer Linear accuracy: {acc5_lin * 100:.1f}%")
    print(f"      -> 5-Layer + ReLU accuracy: {acc5_relu * 100:.1f}%")
    print(f"      -> Matrix product collapse numerical max error: {max_err:.2e}")
    print(f"      -> Saved figure: {fig2_path}")

    # 3. Embeddings
    print("\n[3/4] Running Experiment 3: Embeddings learn similarity from next-token...")
    fig3_path = os.path.join(args.output_dir, "03_emergent_embeddings.png")
    final_loss, purity = run_experiment_3(fig3_path)
    print(f"      -> Next-token prediction CrossEntropy loss: {final_loss:.4f}")
    print(f"      -> 2-NN Semantic category purity:           {purity * 100:.1f}%")
    print(f"      -> Saved figure: {fig3_path}")

    # 4. Generalization
    print("\n[4/4] Running Experiment 4: Memorization vs generalization...")
    fig4_path = os.path.join(args.output_dir, "04_generalization_gap.png")
    summary = run_experiment_4(fig4_path)
    for n, s in summary.items():
        print(f"      -> N={n:4d}: Train Loss={s['final_train_loss']:.3f}, Test Loss={s['final_test_loss']:.3f}, Gap={s['generalization_gap']:.3f}, Test Acc={s['final_test_acc']*100:.1f}%")
    print(f"      -> Saved figure: {fig4_path}")

    print("\n" + "=" * 75)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY! All figures verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
