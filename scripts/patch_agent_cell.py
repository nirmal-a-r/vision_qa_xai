"""Replace the placeholder agent cell with a live episode + LLM narration."""
import json, io, sys

NB = "notebook/VisionQA_RiskControlled_Inspection.ipynb"

MD_AGENT = r"""## 10 · Closed-loop inspection agent

The agent runs the loop a plant actually executes: route each part under the
certified policy, watch the score stream for drift, recalibrate or escalate when
drift fires, and spend a limited labelling budget on the parts that most tighten
the certificate.

**Where the language model sits, and why it sits there.** Every decision below —
accept, review, reject, recalibrate, escalate — is made by the conformal
controller, never by the LLM. A language model cannot carry a statistical
certificate; put one in the decision path and you get fluent prose attached to an
unbounded error rate, which is the exact failure this project argues against. So
Qwen is given the one job it is genuinely good at: turning certified machine
state into a shift report an operator can act on.

The separation is testable. If the model is offline, wrong, or hallucinating, the
inspection outcome is byte-identical — only the prose changes. The cell prints
which backend answered so that claim is visible rather than asserted.
"""

CODE_AGENT = r'''# ---- live closed-loop episode on real cached predictions --------------------
from src.agentic.inspector import InspectionAgent, run_episode
from src.agentic import llm_narrator as NARR

AGENT_DS = "kolektor"      # 89% clean parts -> the only regime where auto-accept is meaningful
AGENT_ALPHA = 0.10

cal_p = f"runs/preds/{AGENT_DS}_yolov8s_cal.json"
test_p = f"runs/preds/{AGENT_DS}_yolov8s_test.json"

if not (os.path.exists(cal_p) and os.path.exists(test_p)):
    print(f"prediction cache missing for {AGENT_DS}; run scripts/run_all_experiments.py")
else:
    cal_rec = json.load(open(cal_p))["records"]
    test_rec = json.load(open(test_p))["records"]

    agent = InspectionAgent(cal_rec, alpha=AGENT_ALPHA, drift_window=150)
    print(f"calibrated on {len(cal_rec)} held-out parts -> lambda={agent.state.lam:.4f} "
          f"certified={agent.state.certified}")

    run_episode(agent, test_rec)
    summ = agent.summary() if callable(getattr(agent, "summary", None)) else agent.state.as_dict()
    st = summ if isinstance(summ, dict) else agent.state.as_dict()

    print(f"\nepisode over {st.get('n_parts', len(test_rec))} parts")
    for k in ("n_accept", "n_review", "n_reject", "n_escapes", "n_defective_seen",
              "drift_alarms", "recalibrations", "labels_requested"):
        if k in st:
            print(f"  {k:20s} {st[k]}")
    for k in ("review_load", "auto_rate", "escape_rate"):
        if k in st:
            print(f"  {k:20s} {st[k]:.4f}")
    print(f"  escape budget alpha  {AGENT_ALPHA}  -> "
          f"{'WITHIN BUDGET' if st.get('escape_rate', 1) <= AGENT_ALPHA else 'OVER BUDGET'}")

    # ---- decision trace: first parts through the loop -----------------------
    if getattr(agent, "log", None):
        print("\nfirst 8 decisions:")
        for e in agent.log[:8]:
            print("  ", json.dumps(e)[:150])

    # ---- LLM narration over certified state ---------------------------------
    print("\n" + "=" * 70)
    ns = NARR.status()
    print(f"LLM backend : {ns['selected_model'] or 'none'}  (mode={ns['mode']})")
    print(f"ollama up   : {ns['ollama_running']}")
    print("=" * 70)

    stats = {"dataset": AGENT_DS, "alpha": AGENT_ALPHA,
             "n_parts": st.get("n_parts"), "auto_accept": st.get("n_accept"),
             "human_review": st.get("n_review"), "auto_reject": st.get("n_reject"),
             "escapes": st.get("n_escapes"), "drift_alarms": st.get("drift_alarms"),
             "recalibrations": st.get("recalibrations"),
             "review_load": round(st.get("review_load", 0), 4),
             "escape_rate": round(st.get("escape_rate", 0), 4)}

    rep = NARR.shift_report(stats)
    print(f"\n--- SHIFT REPORT  (llm={rep['llm']}, source={rep['source']}) ---")
    print(rep["text"])

    # A routed part, explained to the operator
    routed = next((r for r in test_rec
                   if agent._max_score(r) >= agent.state.lam), None)
    if routed is not None:
        rec_view = {"image_id": routed.get("image_id"),
                    "max_score": round(agent._max_score(routed), 4),
                    "n_detections": len(routed.get("pred_scores", []))}
        pol = {"lambda_lo": round(agent.state.lam, 4), "phi": agent.phi}
        ex = NARR.explain_routing(rec_view, pol)
        print(f"\n--- WHY THIS PART WAS ROUTED  (llm={ex['llm']}) ---")
        print(ex["text"])

    print("\nNote: the numbers above come from the certified controller. The LLM only")
    print("describes them - disabling it changes the prose, not a single decision.")
'''


def main():
    nb = json.load(io.open(NB, encoding="utf-8"))
    hit = 0
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"])
        if c["cell_type"] == "markdown" and s.strip().startswith("## 10"):
            nb["cells"][i]["source"] = MD_AGENT.splitlines(keepends=True)
            hit += 1
        elif c["cell_type"] == "code" and 'runs/agent_log.json' in s:
            nb["cells"][i]["source"] = CODE_AGENT.splitlines(keepends=True)
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
            hit += 1
    if hit != 2:
        print(f"WARNING: patched {hit}/2 cells", file=sys.stderr)
    json.dump(nb, io.open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"patched {hit} cells in {NB}")


if __name__ == "__main__":
    main()
