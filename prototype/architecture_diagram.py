import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


FIGS_DIR = Path(__file__).parent / "figures"


def draw_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    colors = {
        "input": "#E8F5E9",
        "short": "#BBDEFB",
        "mid": "#C5CAE9",
        "long": "#E1BEE7",
        "ising": "#FFF9C4",
        "readout": "#FFCCBC",
        "regime": "#F8BBD0",
        "output": "#B2DFDB",
        "feedback": "#FFECB3",
        "arrow": "#37474F",
    }

    def draw_box(x, y, w, h, label, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="#37474F", linewidth=1.5)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, label="", style="->", color="#37474F", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, color=color, lw=lw))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=7,
                    color="#616161", style="italic")

    # Title
    ax.text(8, 9.6, "VolQRC: Quantum Reservoir Computing for Volatility Forecasting",
            ha="center", fontsize=14, fontweight="bold")

    # Input features
    features = ["log(RV_t)", "RV_d", "RV_w", "RV_m", "log-return"]
    draw_box(0.3, 7.5, 2.5, 1.5, "Input Features\n" + "\n".join(features),
             colors["input"], fontsize=8, bold=True)

    # Angle encoding
    draw_box(0.3, 5.8, 2.5, 1.2, "Angle Encoding\nRy(arcsin(x_t))\nα_s, α_m, α_l scaling",
             colors["input"], fontsize=8, bold=True)

    draw_arrow(1.55, 7.5, 1.55, 7.0, "normalize")

    # Onion reservoir bands
    # Short band
    draw_box(4, 8.2, 3, 1.2, "Short Band (⌊N/4⌋ qubits)\nα_s = 1.0 (fast response)\nCaptures recent shocks",
             colors["short"], fontsize=8, bold=True)

    # Mid band
    draw_box(4, 6.6, 3, 1.2, "Mid Band (⌊N/3⌋ qubits)\nα_m = 0.6 (moderate)\nWeekly dynamics",
             colors["mid"], fontsize=8, bold=True)

    # Long band
    draw_box(4, 5.0, 3, 1.2, "Long Band (remainder qubits)\nα_l = 0.3 (slow evolution)\nMonthly dynamics + memory",
             colors["long"], fontsize=8, bold=True)

    draw_arrow(2.8, 6.4, 4.0, 8.6, "short feat")
    draw_arrow(2.8, 6.2, 4.0, 7.0, "mid feat")
    draw_arrow(2.8, 6.0, 4.0, 5.4, "long feat")

    # Ising Hamiltonian
    draw_box(8.5, 6.4, 3, 2.6, "Transverse-Field Ising\nHamiltonian Evolution\n\nH = −Σ Jᵢⱼ σᵢᶻσⱼᶻ − h Σ σᵢˣ\n\nJᵢⱼ ~ N(0,1), fixed\nTrotterized: 4 steps\nAll-to-all coupling via RZZ",
             colors["ising"], fontsize=8, bold=True)

    draw_arrow(7.0, 8.6, 8.5, 8.0, "")
    draw_arrow(7.0, 7.2, 8.5, 7.7, "")
    draw_arrow(7.0, 5.6, 8.5, 7.2, "")

    # Memory feedback loop
    draw_box(8.5, 4.0, 3, 1.6, "Memory Re-injection\n\nMeasure long-band <σᵢᶻ>\n→ feedback as Ry angles\non long-band qubits\n(next time step)",
             colors["feedback"], fontsize=8, bold=True)

    # Feedback arrow (looping back)
    ax.annotate("", xy=(7.0, 5.4), xytext=(8.5, 5.0),
                arrowprops=dict(arrowstyle="->", color="#E65100", lw=2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(7.5, 4.3, "feedback\nloop", fontsize=7, color="#E65100", ha="center", style="italic")

    draw_arrow(11.5, 7.5, 11.5, 5.6, "measure")

    # Observables extraction
    draw_box(12.5, 6.5, 3, 2.2, "Observable Extraction\n\nSingle-qubit: <σᵢᶻ>\nTwo-qubit: <σᵢᶻσⱼᶻ>\n\nN=5 → 15 features\nN=10 → 55 features\nN=20 → 210 features",
             colors["readout"], fontsize=8, bold=True)

    draw_arrow(11.5, 7.5, 12.5, 7.5, "")

    # Classical readout
    draw_box(0.3, 3.0, 3.5, 1.4, "Ridge Regression Readout\nŵ = (XᵀX + λI)⁻¹Xᵀy\n\nInput: [reservoir obs, HAR features]",
             colors["readout"], fontsize=8, bold=True)

    # Regime classifier
    draw_box(4.5, 3.0, 3.5, 1.4, "Quantum Kernel Regime\nClassifier\n\nK(xᵢ,xⱼ) = |<φ(xᵢ)|φ(xⱼ)>|²\nSVM: calm / elevated / crisis",
             colors["regime"], fontsize=8, bold=True)

    # Regime gating
    draw_box(8.5, 3.0, 3, 1.4, "Regime-Gated\nPrediction\n\nRegime label → one-hot\nappended to ridge input",
             colors["regime"], fontsize=8, bold=True)

    draw_arrow(14.0, 6.5, 14.0, 3.7, "")
    ax.annotate("", xy=(3.8, 3.7), xytext=(14.0, 3.7),
                arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=1.5))
    ax.text(8.8, 3.85, "observable vector", fontsize=7, ha="center", color="#616161", style="italic")

    draw_arrow(3.8, 3.4, 4.5, 3.4, "")
    draw_arrow(8.0, 3.7, 8.5, 3.7, "")

    # Final output
    draw_box(12.5, 3.0, 3, 1.4, "Volatility Forecast\n\nlog(RV_{t+1})\n+ Regime label",
             colors["output"], fontsize=9, bold=True)

    draw_arrow(11.5, 3.7, 12.5, 3.7, "")

    # Legend
    legend_items = [
        ("Quantum processing", colors["ising"]),
        ("Multi-scale reservoir", colors["short"]),
        ("Classical readout", colors["readout"]),
        ("Regime classification", colors["regime"]),
        ("Memory feedback", colors["feedback"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        rect = mpatches.FancyBboxPatch((0.3 + i * 3.0, 0.3), 0.4, 0.4,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="#37474F")
        ax.add_patch(rect)
        ax.text(0.9 + i * 3.0, 0.5, label, fontsize=7, va="center")

    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / "volqrc_architecture.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS_DIR / "volqrc_architecture.pdf", bbox_inches="tight")
    print(f"Architecture diagram saved to {FIGS_DIR / 'volqrc_architecture.png'}")
    plt.close(fig)


def draw_onion_allocation_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, n_qubits in zip(axes, [5, 10]):
        from onion_qrc import allocate_onion
        alloc = allocate_onion(n_qubits)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"N = {n_qubits} qubits", fontsize=13, fontweight="bold")

        radii = [0.5, 1.0, 1.5]
        band_colors = ["#BBDEFB", "#C5CAE9", "#E1BEE7"]
        band_labels = [
            f"Short\n({alloc.short_n} q)",
            f"Mid\n({alloc.mid_n} q)",
            f"Long\n({alloc.long_n} q)",
        ]
        counts = [alloc.short_n, alloc.mid_n, alloc.long_n]

        for r, c, label, count in zip(radii, band_colors, band_labels, counts):
            circle = plt.Circle((0, 0), r, facecolor=c, edgecolor="#37474F", linewidth=1.5, alpha=0.7)
            ax.add_patch(circle)

        ax.text(0, 0, band_labels[0], ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(0, 0.75, band_labels[1], ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(0, 1.25, band_labels[2], ha="center", va="center", fontsize=9, fontweight="bold")

        # Draw qubit dots
        for band_idx, (radius, qs) in enumerate(zip([0.35, 0.8, 1.3], [alloc.short_qubits, alloc.mid_qubits, alloc.long_qubits])):
            angles = np.linspace(0, 2 * np.pi, len(qs), endpoint=False)
            for q, angle in zip(qs, angles):
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                ax.plot(x, y, "ko", markersize=5)
                ax.text(x + 0.12, y + 0.12, f"q{q}", fontsize=6, color="#616161")

    fig.suptitle("Onion QRC Band Allocation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / "onion_allocation.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS_DIR / "onion_allocation.pdf", bbox_inches="tight")
    print(f"Onion allocation diagram saved to {FIGS_DIR / 'onion_allocation.png'}")
    plt.close(fig)


def draw_circuit_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    n = 5
    qubit_labels = ["q₀ (S)", "q₁ (M)", "q₂ (L)", "q₃ (L)", "q₄ (L)"]
    y_positions = [5, 4, 3, 2, 1]

    for label, y in zip(qubit_labels, y_positions):
        ax.plot([0.5, 13.5], [y, y], "k-", linewidth=0.8, alpha=0.3)
        ax.text(0.2, y, label, fontsize=9, va="center", ha="right", fontweight="bold")

    # Encoding block
    for i, (y, label) in enumerate(zip(y_positions, ["Ry(α_s·θ)", "Ry(α_m·θ)", "Ry(α_l·θ)", "Ry(α_l·θ)", "Ry(α_l·θ)"])):
        x = 1.5
        ax.plot(x, y, "s", color="#4CAF50", markersize=12)
        ax.text(x, y - 0.35, label, fontsize=6, ha="center", color="#2E7D32")

    ax.text(1.5, 5.6, "Encoding", fontsize=10, ha="center", fontweight="bold", color="#2E7D32")

    # Memory feedback
    x_fb = 2.5
    for i, y in enumerate(y_positions[2:]):
        ax.plot(x_fb, y, "D", color="#FF9800", markersize=10)
    ax.text(x_fb, 5.6, "Memory\nFeedback", fontsize=8, ha="center", fontweight="bold", color="#E65100")

    # Trotter step (repeated)
    for step in range(4):
        x_start = 3.5 + step * 2.3

        # Transverse field (Rx)
        for y in y_positions:
            ax.plot(x_start, y, "o", color="#2196F3", markersize=10)
        ax.text(x_start, 5.6, "Rx(h·dt)", fontsize=7, ha="center", color="#1565C0")

        # ZZ couplings (draw a few representative ones)
        x_zz = x_start + 0.8
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                y1 = y_positions[i_idx]
                y2 = y_positions[j_idx]
                if j_idx == i_idx + 1:
                    ax.plot([x_zz, x_zz], [y1, y2], color="#9C27B0", linewidth=2)
                    ax.plot(x_zz, y1, "o", color="#9C27B0", markersize=6)
                    ax.plot(x_zz, y2, "o", color="#9C27B0", markersize=6)

        ax.text(x_zz, 5.6, f"RZZ(J·dt)\nStep {step + 1}", fontsize=7, ha="center", color="#6A1B9A")

    # Measurement
    x_meas = 13
    for y in y_positions:
        ax.plot(x_meas, y, "o", color="#F44336", markersize=12)
        ax.plot([x_meas - 0.15, x_meas + 0.15], [y + 0.15, y - 0.15], "r-", linewidth=1.5)
        ax.plot([x_meas - 0.15, x_meas + 0.15], [y - 0.15, y + 0.15], "r-", linewidth=1.5)

    ax.text(x_meas, 5.6, "Measure\n<σᵢᶻ>, <σᵢᶻσⱼᶻ>", fontsize=8, ha="center", fontweight="bold", color="#C62828")

    fig.suptitle("VolQRC Circuit: Single Time Step (N=5)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_DIR / "circuit_diagram.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS_DIR / "circuit_diagram.pdf", bbox_inches="tight")
    print(f"Circuit diagram saved to {FIGS_DIR / 'circuit_diagram.png'}")
    plt.close(fig)


if __name__ == "__main__":
    draw_architecture_diagram()
    draw_onion_allocation_diagram()
    draw_circuit_diagram()
    print("All diagrams generated successfully.")
