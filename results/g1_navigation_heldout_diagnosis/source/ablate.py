import sys, json
sys.path.insert(0, r"C:\Users\24adi\OneDrive\DREAM")
import experiments.g1_navigation_eval as E
from dream.tools import mobile as M
variant = sys.argv[1]
if variant == "no_revisit_filter":
    M.MobileToolbox._visited = lambda self, xy, heading: False
kind, seed = sys.argv[2], int(sys.argv[3])
r = E.episode((kind, seed, "rpp", {"time_budget_s":240,"max_steps":60}, "ablation"))
print(json.dumps({"variant": variant, "kind": kind, "seed": seed, "success": r["success"], "sim_s": round(r["sim_s"],1), "reason": r["reason"][:70], "contact": r["contact_steps"]}))
