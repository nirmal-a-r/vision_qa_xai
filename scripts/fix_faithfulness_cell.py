"""Point the faithfulness section at the files that actually exist.

The cell looked for `runs/faithfulness_results.json`, a name nothing ever
writes. `scripts/compute_faithfulness.py` writes one file per dataset,
`runs/faithfulness_<dataset>.json`, so section 8 rendered its "not populated"
placeholder even when the data was sitting on disk.
"""
import json, io, glob, sys

NB = "notebook/VisionQA_RiskControlled_Inspection.ipynb"

CODE = r'''import glob  # figures.py exposes OKABE, not a C dict; glob is not in the setup cell
# Audited faithfulness. compute_faithfulness.py writes one file per dataset,
# runs/faithfulness_<dataset>.json, each a list of per-image records.
fp_list = sorted(glob.glob("runs/faithfulness_*.json"))
if not fp_list:
    print("no faithfulness cache; run:  python scripts/compute_faithfulness.py 25 neu")
else:
    frames = []
    for fp in fp_list:
        ds = os.path.basename(fp).replace("faithfulness_", "").replace(".json", "")
        rows = json.load(open(fp))
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d.insert(0, "dataset", ds)
        frames.append(d)

    df_f = pd.concat(frames, ignore_index=True)
    n_img = len(df_f)

    summary = (df_f.groupby("dataset")
               .agg(images=("faithfulness", "size"),
                    insertion_auc=("insertion_auc", "mean"),
                    energy_pointing=("energy_pointing", "mean"),
                    chance_level=("chance_level", "mean"),
                    pointing_game=("pointing_game", "mean"),
                    faithfulness=("faithfulness", "mean"))
               .reset_index())
    # Energy pointing is only meaningful against the chance level set by how much
    # of the frame the boxes cover; the raw number alone says nothing.
    summary["lift_over_chance"] = summary.energy_pointing / summary.chance_level
    print(summary.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(df_f.faithfulness, bins=20, color=F.OKABE[0], edgecolor="white")
    axes[0].set_title("per-image faithfulness"); axes[0].set_xlabel("score")

    axes[1].scatter(df_f.chance_level, df_f.energy_pointing, s=26,
                    color=F.OKABE[0], alpha=.75, edgecolor="white", linewidth=.6)
    lim = [0, max(df_f.chance_level.max(), df_f.energy_pointing.max()) * 1.08]
    axes[1].plot(lim, lim, ls="--", lw=1.2, color="#888888")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("chance level (box area fraction)")
    axes[1].set_ylabel("energy pointing")
    axes[1].set_title("above the diagonal = better than chance")

    axes[2].scatter(df_f.insertion_auc, df_f.faithfulness, s=26,
                    color=F.OKABE[1], alpha=.75, edgecolor="white", linewidth=.6)
    axes[2].set_xlabel("insertion AUC"); axes[2].set_ylabel("composite faithfulness")
    axes[2].set_title("causal vs localisation agreement")
    for a in axes:
        a.grid(alpha=.25)
    fig.suptitle(f"Explanation faithfulness audit  ({n_img} images, "
                 f"{len(summary)} dataset(s))", y=1.04)
    fig.tight_layout()
    F.save(fig, "fig14_faithfulness"); plt.show()

    above = (df_f.energy_pointing > df_f.chance_level).mean()
    print(f"\n{above*100:.0f}% of maps concentrate more attribution inside the "
          f"defect boxes than box area alone would give.")
    print("Faithfulness is what gates the triage policy in section 9, which is why")
    print("it is measured per image rather than reported as a dataset average.")
'''


def main():
    nb = json.load(io.open(NB, encoding="utf-8"))
    hit = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "faithfulness_results.json" in s:
            nb["cells"][i]["source"] = CODE.splitlines(keepends=True)
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
            hit += 1
    # the placeholder messages name scripts that do not exist
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        s2 = s.replace("run scripts/run_triage.py to populate this section",
                       "run scripts/run_all_experiments.py to populate this section")
        s2 = s2.replace("scripts/run_risk_experiment.py --drift",
                        "scripts/run_all_experiments.py")
        if s2 != s:
            nb["cells"][i]["source"] = s2.splitlines(keepends=True)
            hit += 1
    json.dump(nb, io.open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"patched {hit} cell(s)")


if __name__ == "__main__":
    main()
