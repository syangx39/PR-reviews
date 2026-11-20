#!/usr/bin/env python3
"""Check every number in torchtitan/168/README.md against the committed summaries (Python 3 stdlib only).

Each test names the README section it grounds. Run from anywhere:
    python3 torchtitan/168/tools/verify_pr168_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUM = ROOT / "summaries"
ARMS = ("A_sel", "C3_ft", "C2_tf", "C4_tt")


def load(name: str):
    return json.loads((SUM / name).read_text(encoding="utf-8"))


def graph(arm: str, layer: str, stage: str, row: int) -> dict:
    for g in load("aot_graph_stats.json")["arms"][arm]:
        if (g["layer_type"], g["stage"], g["row_bucket"]) == (layer, stage, row):
            return g
    raise KeyError((arm, layer, stage, row))


class Setup(unittest.TestCase):
    """README section 1."""

    def test_overlay_sha256(self) -> None:
        digest = hashlib.sha256((ROOT / "patches" / "activation_checkpoint_pr168.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "3d4fd13f8c554fc97783728a2ec4efbfa763931bdf0da2eaf970fef5a46c9a55")
        self.assertEqual(load("run_setup.json")["torchtitan"]["pr168_overlay_sha256"], digest)

    def test_recipe(self) -> None:
        s = load("run_setup.json")
        env = s["recipe_env"]
        self.assertEqual((env["NNODES_TRAINER"], env["N_CHIPS_TRAINER"]), ("4", "8"))  # 4 hosts x 8 devices = 32 devices = 16 chips
        self.assertEqual((env["NNODES_ROLLOUT"], env["N_CHIPS_ROLLOUT"]), ("4", "8"))
        self.assertEqual((env["TENSOR_PARALLEL_SIZE"], env["CONTEXT_PARALLEL_SIZE"]), ("1", "1"))
        self.assertEqual(env["ROLLOUT_TENSOR_PARALLEL_SIZE"], "2")  # 32 devices / TP2 = 16 engines
        self.assertEqual((env["TRAIN_BATCH_SIZE"], env["ROLLOUT_N"]), ("128", "16"))
        self.assertEqual((env["MAX_PROMPT_LEN"], env["MAX_RESPONSE_LEN"]), ("256", "4096"))
        self.assertEqual((env["LEARNING_RATE"], env["LR_WARMUP_STEPS"], env["TOTAL_TRAINING_STEPS"]), ("2.5e-6", "10", "30"))
        self.assertEqual(env["VAL_BATCH_SIZE"], "1319")
        ov = s["recipe_overrides"]
        self.assertIn("actor_rollout_ref.actor.torchtitan.activation_checkpoint=selective", ov)
        self.assertIn("++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.model_kernels=none", ov)
        self.assertIn("+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.fixed_shape_rows=8192:14336:16384", ov)
        self.assertEqual(s["model_and_data"]["hf_revision"][:8], "2fc06364")
        self.assertEqual(s["torchtitan"]["in_image_commit"][:8], "aaf742e4")
        self.assertEqual(s["runtime_packages"], {"python": "3.12.12", "torch": "2.13.0+cpu", "torch_tpu": "0.2.0",
                                                 "libtpu": "0.0.50", "vllm": "0.32.0", "jax": "0.12.0.1"})
        self.assertTrue(any("#131" in x and "eager" in x for x in s["verl_hardware_plugin"]["included_prs"]))
        self.assertTrue(any("#130" in x for x in s["verl_hardware_plugin"]["included_prs"]))

    def test_replay_same_hosts_and_policy_audit(self) -> None:
        hosts = set()
        for arm in ARMS:
            w = load(f"rank0_warm_timings_{arm}.json")
            hosts.add(w["hw_audit"]["tpu_worker_hostnames"])
            self.assertEqual(w["hw_audit"]["tpu_topology"], "2x2x4")
            ac = w["ac_audit"]
            self.assertTrue(ac["policy_cls"].endswith("components.activation_checkpoint.TPUSelectiveAC"))
            rb, sp = arm in ("C2_tf", "C4_tt"), arm in ("C3_ft", "C4_tt")
            self.assertEqual((ac["recompute_bmm"], ac["save_pallas_attention"]), (rb, sp))
            self.assertEqual((ac["bmm_in_save_ops"], ac["splash_in_save_ops"]), (not rb, sp))
            self.assertTrue(ac["collectives_excluded"])
        self.assertEqual(len(hosts), 1)

    def test_multiseed_runs_audit_and_hosts(self) -> None:
        runs = load("multiseed_comparison.json")["runs"]
        main = [f"{a}_s{s}" for a in ("a_sel", "c4_tt") for s in (1, 42, 101)]
        trainer_sets = set()
        for r in main:
            line = runs[r]["ac_audit_line"]
            c4 = r.startswith("c4")
            self.assertIn(f"recompute_bmm={c4}", line)
            self.assertIn(f"save_pallas_attention={c4}", line)
            self.assertEqual(runs[r]["tpu_sync_parity_exact_replicas"], 16)  # 16 rollout engines
            trainer_sets.add(tuple(runs[r]["trainer_nodes"]))
        self.assertEqual(len(trainer_sets), 6)  # each 30-step run on its own trainer hosts


class Replay(unittest.TestCase):
    """README section 2 table and notes."""

    def test_batch_and_layout(self) -> None:
        ab = load("ab_compile_token_summary.json")["stage1_replay"]
        for arm in ARMS:
            self.assertEqual(ab[arm]["real_tokens"], 2328294)
            self.assertEqual(ab[arm]["layout_rows"], [14336] * 4 + [8192] * 2)
            self.assertEqual(ab[arm]["xprof_module_launches_per_step"], 836)
            self.assertEqual(load("xprof_7cat_comparison.json")["summaries"][arm]["num_xla_modules"], 836)
            self.assertEqual(len(load(f"rank0_warm_timings_{arm}.json")["warm_iters"]), 5)
        self.assertTrue(load("stage1_replay_comparison.json")["input_batch_sha256"].startswith("e3d67e4f"))

    def test_time_and_hbm(self) -> None:
        ab = load("ab_compile_token_summary.json")["stage1_replay"]
        expect = {"A_sel": (5.1302, 24.10), "C3_ft": (5.0412, 24.43), "C2_tf": (4.9796, 19.70), "C4_tt": (4.9099, 20.06)}
        for arm, (fb, hbm) in expect.items():
            self.assertAlmostEqual(ab[arm]["fwd_bwd_s"], fb, places=4)
            self.assertAlmostEqual(ab[arm]["peak_hbm_gb"], hbm, delta=0.005)
            w = load(f"rank0_warm_timings_{arm}.json")["warm_iters"]
            # fwd_bwd = mean of timed iterations 2-5 on rank 0; HBM = max over timed iterations on rank 0
            self.assertAlmostEqual(statistics.mean(x["fwd_bwd_s"] for x in w[1:]), fb, places=4)
            self.assertAlmostEqual(max(x["max_memory_allocated_gb"] for x in w), ab[arm]["peak_hbm_gb"], places=4)
        a = ab["A_sel"]
        for arm, dt, pct, dh in (("C3_ft", -0.089, -1.7, 0.32), ("C2_tf", -0.151, -2.9, -4.40), ("C4_tt", -0.220, -4.3, -4.05)):
            self.assertAlmostEqual(ab[arm]["fwd_bwd_s"] - a["fwd_bwd_s"], dt, delta=0.0006)
            self.assertAlmostEqual(100 * (ab[arm]["fwd_bwd_s"] / a["fwd_bwd_s"] - 1), pct, delta=0.05)
            self.assertAlmostEqual(ab[arm]["peak_hbm_gb"] - a["peak_hbm_gb"], dh, delta=0.006)
        self.assertAlmostEqual(100 * (ab["C4_tt"]["peak_hbm_gb"] / a["peak_hbm_gb"] - 1), -16.8, delta=0.05)

    def test_saved_tensors_and_splash_calls(self) -> None:
        for arm in ARMS:
            rb, sp = arm in ("C2_tf", "C4_tt"), arm in ("C3_ft", "C4_tt")
            g = graph(arm, "gdn", "forward", 14336)
            self.assertEqual(g["bmm_calls"], 673)
            self.assertEqual((g["num_saved_tensors"], round(g["saved_tensors_mib"], 2)), (26, 269.24) if rb else (699, 604.74))
            self.assertEqual(graph(arm, "gdn", "forward", 8192)["num_saved_tensors"], 26 if rb else 411)
            f = graph(arm, "full_attn", "forward", 14336)
            self.assertEqual((f["num_saved_tensors"], round(f["saved_tensors_mib"], 2)), (20, 312.86) if sp else (18, 256.42))
            b = graph(arm, "full_attn", "backward", 14336)
            self.assertEqual(b["splash_fwd_calls"] + b["splash_bwd_calls"], 1 if sp else 2)
            self.assertEqual(b["splash_bwd_calls"], 1)
        # PR formula 27 + 3 * seq_len / 64
        self.assertEqual(27 + 3 * 14336 // 64, 699)
        self.assertEqual(27 + 3 * 8192 // 64, 411)
        self.assertAlmostEqual((604.738 - 269.238) * 18 / 1024, 5.9, delta=0.01)
        self.assertAlmostEqual(312.86 - 256.42, 56.44, places=2)

    def test_pg_loss_grad_norm(self) -> None:
        pw = load("noise_floor_comparison.json")["pairwise_comparisons"]
        c3 = pw["C3_ft_vs_A_sel_stage1"]
        self.assertEqual((round(c3["pg_loss_base"], 6), round(c3["grad_norm_base"], 6)), (0.090924, 0.333176))
        self.assertEqual((round(c3["pg_loss_arm"], 6), round(c3["grad_norm_arm"], 6)), (0.090924, 0.33317))
        c4c2 = pw["C4_tt_vs_C2_tf_stage1"]
        for side in ("base", "arm"):
            self.assertEqual((round(c4c2[f"pg_loss_{side}"], 6), round(c4c2[f"grad_norm_{side}"], 6)), (0.090961, 0.332724))

    def test_fusions_and_xprof(self) -> None:
        h = load("hlo_fx_comparison_14336.json")["arms"]
        self.assertEqual(h["A_sel"]["programs"]["gdn_backward"]["hlo_fusion_total"], 2286)
        self.assertEqual(h["C4_tt"]["programs"]["gdn_backward"]["hlo_fusion_total"], 2833)
        x = load("xprof_7cat_comparison.json")["summaries"]
        fa = {a: x[a]["breakdown_s"]["backward"]["full_attention"] for a in ARMS}
        self.assertEqual((round(fa["A_sel"], 4), round(fa["C3_ft"], 4)), (0.4982, 0.4276))
        self.assertAlmostEqual(fa["A_sel"] - fa["C3_ft"], 0.0706, delta=0.00005)
        self.assertAlmostEqual(fa["A_sel"] - fa["C4_tt"], 0.0715, delta=0.00006)


class Numerics(unittest.TestCase):
    """README section 2.1."""

    def test_noise_floor_table(self) -> None:
        pw = load("noise_floor_comparison.json")["pairwise_comparisons"]

        def row(k):
            v = pw[k]
            return v["pg_loss_abs_diff"], v["grad_norm_rel_diff"], v["delta_w"]

        pl, gn, dw = row("A_sel_rep1_vs_A_sel_rep2_same_slice")
        self.assertEqual((pl, gn), (0.0, 7.16e-6))
        self.assertEqual((round(dw["cosine"], 6), round(100 * dw["sign_agreement"], 2)), (0.999938, 99.97))
        self.assertAlmostEqual(dw["max_abs_diff"], 2.89e-7, delta=5e-10)
        for k in ("C4_tt_rep1_vs_C4_tt_rep2_same_slice", "C4_tt_rep1_vs_C4_tt_stage1", "C4_tt_rep2_vs_C4_tt_stage1", "C4_tt_vs_C2_tf_stage1"):
            pl, gn, dw = row(k)
            self.assertEqual((pl, gn, dw["cosine"], dw["sign_agreement"], dw["max_abs_diff"]), (0.0, 0.0, 1.0, 1.0, 0.0))
        pl, gn, dw = row("C3_ft_vs_A_sel_stage1")
        self.assertEqual((pl, round(gn, 7)), (0.0, 1.79e-5))
        self.assertEqual((round(dw["cosine"], 6), round(100 * dw["sign_agreement"], 2)), (0.99996, 99.98))
        self.assertAlmostEqual(dw["max_abs_diff"], 3.20e-7, delta=5e-10)
        for k, cos, sign, g in (("A_sel_rep1_vs_C4_tt_rep1_same_slice", 0.988949, 99.21, 1.35e-3),
                                ("A_sel_rep2_vs_C4_tt_rep2_same_slice", 0.988937, 99.20, 1.36e-3)):
            pl, gn, dw = row(k)
            self.assertAlmostEqual(pl, 3.72e-5, delta=5e-8)
            self.assertAlmostEqual(gn, g, delta=5e-6)
            self.assertEqual((round(dw["cosine"], 6), round(100 * dw["sign_agreement"], 2)), (cos, sign))
            self.assertAlmostEqual(dw["max_abs_diff"], 4.99e-7, delta=5e-10)

    def test_pass_criteria(self) -> None:
        s1 = load("stage1_replay_comparison.json")
        c = s1["pass_criteria"]
        self.assertEqual((c["pg_loss_abs_diff_max"], c["grad_norm_rel_diff_max"], c["delta_w_cosine_min"]), (1e-4, 1e-3, 0.9999))
        arms = s1["arms"]
        self.assertTrue(arms["C3_ft"]["passed"])
        for a in ("C2_tf", "C4_tt"):
            self.assertFalse(arms[a]["passed"])
            self.assertGreater(arms[a]["grad_norm_rel_diff"], c["grad_norm_rel_diff_max"])
            self.assertLess(arms[a]["delta_w"]["cosine"], c["delta_w_cosine_min"])
            self.assertLessEqual(arms[a]["pg_loss_abs_diff"], c["pg_loss_abs_diff_max"])


class MultiSeed(unittest.TestCase):
    """README section 3 and 4."""

    def test_group_table(self) -> None:
        g = load("multiseed_comparison.json")["groups"]
        a, c = g["gsc_A_sel_3seed_matched"], g["gsc_C4_tt_3seed"]
        self.assertEqual(a["runs"], ["a_sel_s1", "a_sel_s42", "a_sel_s101"])
        self.assertEqual(c["runs"], ["c4_tt_s1", "c4_tt_s42", "c4_tt_s101"])

        def ms(grp, k, nd=2):
            return round(grp[k]["mean"], nd), round(grp[k]["sd"], nd)

        self.assertEqual(ms(a, "mean_step_s_excl_ckpt"), (27.24, 0.48))
        self.assertEqual(ms(c, "mean_step_s_excl_ckpt"), (26.96, 0.75))
        self.assertAlmostEqual(c["mean_step_s_excl_ckpt"]["mean"] - a["mean_step_s_excl_ckpt"]["mean"], -0.29, delta=0.005)
        self.assertEqual(ms(a, "mean_update_actor_engine_s"), (4.98, 0.14))
        self.assertEqual(ms(c, "mean_update_actor_engine_s"), (4.83, 0.54))
        self.assertEqual((round(a["engine_mfu_pct"]["mean"], 2), round(c["engine_mfu_pct"]["mean"], 2)), (5.83, 5.99))
        ha, hc = a["max_memory_allocated_gb"]["mean"], c["max_memory_allocated_gb"]["mean"]
        self.assertEqual((round(ha, 2), round(hc, 2), round(hc - ha, 2)), (23.53, 20.04, -3.49))
        self.assertAlmostEqual(100 * (hc / ha - 1), -14.9, delta=0.05)
        self.assertEqual(ms(a["validation"]["30"], "accuracy_pct"), (79.45, 1.0))
        self.assertEqual(ms(c["validation"]["30"], "accuracy_pct"), (78.72, 1.1))

    def test_per_seed_accuracy(self) -> None:
        runs = load("multiseed_comparison.json")["runs"]
        expect = {"a_sel_s1": 79.83, "c4_tt_s1": 78.17, "a_sel_s42": 80.21, "c4_tt_s42": 79.98,
                  "a_sel_s101": 78.32, "c4_tt_s101": 78.01, "a_sel_s1_manifest": 77.71}
        for r, acc in expect.items():
            self.assertEqual(round(runs[r]["validation"]["30"]["accuracy_pct"], 2), acc)
            self.assertEqual(runs[r]["validation"]["30"]["n"], 1319)
        m = runs["a_sel_s1_manifest"]
        self.assertEqual((m["arm"], m["seed"], m["checkpoint_steps"]), ("A_sel_manifest", 1, []))

    def test_mcnemar(self) -> None:
        pm = load("multiseed_comparison.json")["paired_mcnemar"]
        expect = {
            "gsc_seed1_C4_minus_A_matched": (1319, -1.67, -3.77, 0.43),
            "gsc_seed1_C4_minus_A_manifest": (1319, 0.45, -1.66, 2.57),
            "gsc_seed42_C4_minus_A": (1319, -0.23, -2.43, 1.97),
            "gsc_seed101_C4_minus_A": (1319, -0.30, -2.50, 1.89),
            "pooled_3seed_C4_minus_A_matched": (3957, -0.73, -1.98, 0.52),
            "pooled_3seed_C4_minus_A_with_manifest_s1": (3957, -0.03, -1.28, 1.23),
        }
        for k, (n, d, lo, hi) in expect.items():
            s = pm[k]["30"]
            self.assertEqual((s["n"], round(s["diff_b_minus_a_pp"], 2), round(s["ci95_low_pp"], 2), round(s["ci95_high_pp"], 2)), (n, d, lo, hi))
            self.assertTrue(s["ci95_spans_zero"])
        self.assertAlmostEqual(pm["gsc_seed1_A_matched_minus_A_manifest"]["30"]["diff_b_minus_a_pp"], 2.12, delta=0.005)

    def test_buckets_steps_4_30(self) -> None:
        runs = load("multiseed_comparison.json")["runs"]
        for r in [f"{a}_s{s}" for a in ("a_sel", "c4_tt") for s in (1, 42, 101)]:
            self.assertEqual(set(runs[r]["row_bucket_counts_4_30"]), {"8192", "14336"})
            self.assertIn("16384", runs[r]["prewarm_costs_s"])


class Readme(unittest.TestCase):
    def test_no_removed_material(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for s in ("B_none", "B_gdn", "B_attn", "host_gap", "pr2951", "pr2952", "PROMOTE_F32"):
            self.assertNotIn(s, readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
