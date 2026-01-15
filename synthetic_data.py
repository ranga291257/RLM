import argparse
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class SyntheticDocConfig:
    asset: str = "C-201"
    num_entries: int = 250
    start_date: date = date(2025, 1, 1)
    seed: int = 42


def generate_synthetic_doc(cfg: SyntheticDocConfig) -> str:
    """
    Generate a synthetic maintenance log for demos.

    - Seeded randomness makes outputs repeatable.
    - Includes recurring patterns (vibration, lube temp, surge valve, suction DP) + noise events.
    """
    rng = random.Random(cfg.seed)

    alarms = [
        ("VIB_HI_2", "high vibration", [
            "VIB RMS trending up",
            "vibration reduced slightly after tightening coupling guard",
            "correlates with bearing temperature spikes BEAR_TEMP_3",
        ]),
        ("LUBE_OIL_TEMP_HI", "lube oil temperature high", [
            "lube oil cooler outlet temp rising",
            "fan cycling irregular",
            "temperature normalized after cleaning cooler fins",
        ]),
        ("SURGE_CTRL_VALVE_STUCK", "anti-surge valve feedback lag", [
            "valve position feedback oscillated",
            "actuator air supply pressure low",
            "regulator drift on instrument air header",
        ]),
        ("SUCTION_FILTER_DP_HI", "suction filter differential pressure high", [
            "suction filter DP increased",
            "inlet strainer partially blocked",
            "DP back to normal after replacement",
        ]),
        ("BEAR_TEMP_3_HI", "bearing temperature spike", [
            "bearing temperature spikes BEAR_TEMP_3",
            "oil sample metal particles elevated",
            "recommend bearing inspection and alignment verification",
        ]),
    ]

    actions = [
        "Action: reduced load to 70%.",
        "Action: checked lube oil pressure OK.",
        "Action: cleaned cooler fins; verified fan VFD parameters.",
        "Action: replaced inlet strainer; DP normalized.",
        "Action: tightened coupling guard; vibration reduced slightly.",
        "Action: reset instrument air regulator to 6.2 bar.",
        "Action: oil sample taken; sent to lab.",
        "Recommendation: verify alignment at next shutdown.",
        "Recommendation: plan bearing inspection and alignment verification.",
        "Suggested actions: verify cooler fan control loop; add DP monitoring on suction filter.",
    ]

    noise_events = [
        "Operator note: minor seal oil header fluctuation; stabilized after 5 min.",
        "Inspection: found slight oil misting near flange; tightened bolts; no further leak.",
        "Alarm event: 'VFD_COMM_WARN' transient; cleared automatically.",
        "Operator report: suction pressure oscillations during startup; stabilized at normal load.",
        "Maintenance: checked vibration probes; calibration within tolerance.",
    ]

    lines: List[str] = []
    lines.append(f"PLANT MAINTENANCE LOG — {cfg.asset} (SYNTHETIC)\n")

    d = cfg.start_date
    for _ in range(max(1, int(cfg.num_entries))):
        d += timedelta(days=rng.randint(0, 3))

        roll = rng.random()
        if roll < 0.72:
            code, short, evidences = rng.choice(alarms)
            evidence = rng.choice(evidences)
            extra = ""
            if "pressure low" in evidence.lower():
                extra = f" ({rng.uniform(4.8, 5.6):.1f} bar)."
            elif "dp" in evidence.lower():
                extra = f" ({rng.uniform(18, 65):.0f} kPa)."
            elif "temp" in evidence.lower():
                extra = f" ({rng.uniform(85, 110):.0f} C)."

            lines.append(f"[{d.isoformat()}] Alarm event: \"{code}\". {short.title()} on {cfg.asset}.")
            lines.append(f"Evidence: {evidence}{extra}")
            lines.append(rng.choice(actions))
        else:
            lines.append(f"[{d.isoformat()}] {rng.choice(noise_events)}")
            if rng.random() < 0.35:
                lines.append(rng.choice(actions))

        lines.append("")

    lines.append(f"[{(d + timedelta(days=2)).isoformat()}] Root cause note (synthetic):")
    lines.append("Preliminary: misalignment + intermittent lube cooling performance + suction restriction episodes.")
    lines.append("Suggested actions: alignment check; inspect coupling; verify cooler fan control; instrument air audit; add DP monitoring.")
    lines.append("\nEND SYNTHETIC LOG")

    return "\n".join(lines).strip()


def write_synthetic_doc(path: str, cfg: SyntheticDocConfig) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_synthetic_doc(cfg), encoding="utf-8")
    return out


if __name__ == "__main__":
    from pathlib import Path
    
    # Default to input/ directory
    PROJECT_ROOT = Path(__file__).resolve().parent
    INPUT_DIR = PROJECT_ROOT / "input"
    INPUT_DIR.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Generate a synthetic maintenance log text file.")
    parser.add_argument(
        "--out",
        default=str(INPUT_DIR / "synthetic_maintenance_log.txt"),
        help="Output .txt path (default: input/synthetic_maintenance_log.txt).",
    )
    parser.add_argument("--entries", type=int, default=250, help="Number of log entries to generate.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (repeatable output).")
    parser.add_argument("--asset", default="C-201", help="Asset tag, e.g. C-201.")
    args = parser.parse_args()

    cfg = SyntheticDocConfig(asset=args.asset, num_entries=args.entries, seed=args.seed)
    out_path = write_synthetic_doc(args.out, cfg)
    print(f"Wrote synthetic doc to: {out_path.resolve()}")
