"""
Build all figures and manuscript for Nature Methods submission.
Run from project root: python scripts/build_all.py
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(BASE, 'scripts')

SCRIPTS_TO_RUN = [
    'figure1_overview.py',
    'figure2_theoretical.py',
    'figure3_biological.py',
    'figure4_validation.py',
    'figure5_benchmark.py',
    'figure_supplementary.py',
    'generate_manuscript.py',
]


def main():
    print("=" * 60)
    print("  Building Nature Methods Submission Package")
    print("=" * 60)

    for script in SCRIPTS_TO_RUN:
        path = os.path.join(SCRIPTS, script)
        print(f"\n{'=' * 60}")
        print(f"  Running: {script}")
        print(f"{'=' * 60}")
        result = subprocess.run([sys.executable, path], cwd=BASE,
                                capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {script} failed with return code {result.returncode}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Figures: {os.path.join(BASE, 'figures')}")
    print(f"  Manuscript: {os.path.join(BASE, 'submission', 'manuscript_v2.md')}")
    print()


if __name__ == '__main__':
    main()
