"""Extract object->room-type priors and object synonyms from ConceptNet 5.7 and WordNet 3.0.

Source of truth: the downloaded ConceptNet assertions dump (not the API, which was down when
this was written) and the Princeton WordNet 3.0 dict files. Nothing here is tuned to any DREAM
benchmark; the object and place vocabularies below are only the names we need to look up, and
every edge in the output is an edge that exists in the source data.

Run (stdlib only, parse-only, no execution of downloaded content):
    python -I experiments/extract_household_knowledge.py \
        --conceptnet-dump <conceptnet-assertions-5.7.0.csv.gz> --wordnet-dir <.../wordnet> \
        --out-dir third_party/household_knowledge
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

OBJECTS = (
    "refrigerator countertop sink stove oven kitchen_cabinet microwave dishwasher dining_table "
    "chair bed wardrobe nightstand dresser desk office_chair bookshelf sofa coffee_table television "
    "tv_stand armchair toilet washbasin shower bathtub workbench tool_rack storage_shelf "
    "washing_machine dryer chest_freezer filing_cabinet kettle cabinet kitchen_island"
).split()
PLACES = (
    "kitchen bedroom bathroom living_room dining_room office garage workshop laundry_room "
    "storage_room pantry basement hallway closet"
).split()

# Surface-form variants tried in addition to the canonical name. The variant that produced each
# edge is recorded. Chosen from ordinary English synonyms, before looking at any results.
OBJECT_VARIANTS = {
    "refrigerator": ["fridge", "icebox"],
    "countertop": ["counter", "counter_top", "kitchen_counter", "worktop"],
    "sink": ["kitchen_sink"],
    "stove": ["cooker", "range"],
    "kitchen_cabinet": ["kitchen_cupboard", "cupboard"],
    "microwave": ["microwave_oven"],
    "nightstand": ["night_stand", "bedside_table", "night_table"],
    "dresser": ["chest_of_drawers"],
    "bookshelf": ["bookcase", "book_shelf"],
    "sofa": ["couch", "settee"],
    "television": ["tv", "tv_set", "telly"],
    "tv_stand": ["television_stand", "tv_cabinet"],
    "armchair": ["arm_chair"],
    "toilet": ["lavatory", "loo"],
    "washbasin": ["wash_basin", "washbowl", "hand_basin", "handbasin"],
    "bathtub": ["bath_tub", "bath", "tub"],
    "workbench": ["work_bench"],
    "washing_machine": ["washer"],
    "dryer": ["clothes_dryer", "tumble_dryer"],
    "chest_freezer": ["freezer", "deep_freezer"],
    "filing_cabinet": ["file_cabinet"],
    "wardrobe": ["armoire"],
}
PLACE_VARIANTS = {
    "living_room": ["lounge", "sitting_room", "livingroom"],
    "laundry_room": ["laundry", "utility_room"],
    "storage_room": ["storeroom", "store_room"],
    "basement": ["cellar"],
    "hallway": ["hall", "corridor"],
    "bathroom": ["restroom", "bath_room"],
    "office": ["home_office"],
    "dining_room": ["dining_hall"],
}
EDGE_RELS = ("/r/AtLocation", "/r/LocatedNear", "/r/HasA", "/r/PartOf", "/r/RelatedTo")
# Optional sense overrides (object -> WordNet 3.0 noun synset offsets); empty by default: the target
# household-artifact sense is the first artifact synset of the object name.
# Two corrections where the first artifact-tagged synset is not the household object
# (television.n.03 is the broadcast medium; toilet.n.01 is the room).
SENSE_OVERRIDE: dict[str, list[str]] = {
    "television": ["04405907"],  # television receiver
    "toilet": ["04446521"],  # toilet, commode, crapper, pot... (the fixture)
}
SYN_REL = "/r/Synonym"
ARTIFACT_OFFSET = "00021939"  # WordNet 3.0 noun synset artifact.n.01
POS_OK = {"n", None}


def variants(canon: str, table: dict[str, list[str]]) -> list[str]:
    return [canon] + table.get(canon, [])


def build_lookup(vocab, table):
    """Map concept term (underscored lowercase) -> (canonical, variant)."""
    out = {}
    for canon in vocab:
        for v in variants(canon, table):
            out.setdefault(v, (canon, v))
    return out


def parse_concept(uri: str):
    """'/c/en/washing_machine/n/wn/artifact' -> ('en', 'washing_machine', 'n')."""
    p = uri.split("/")
    if len(p) < 4 or p[1] != "c":
        return None
    return p[2], p[3], (p[4] if len(p) > 4 else None)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def edge_source(meta: dict) -> str:
    srcs = meta.get("sources", [])
    parts = []
    for s in srcs:
        parts.append(s.get("contributor") or s.get("activity") or s.get("process") or "?")
    return meta.get("dataset", "?") + " | " + "; ".join(sorted(set(parts)))[:200]


# ---------------------------------------------------------------- WordNet (plain files)
class WordNet:
    def __init__(self, d: Path):
        self.hypo = {}  # offset -> direct hyponym offsets
        self.syn = {}  # offset -> (lemmas, hypernym offsets, gloss)
        for line in (d / "data.noun").read_text(encoding="utf-8").splitlines():
            if line.startswith("  "):
                continue
            head, _, gloss = line.partition("|")
            t = head.split()
            n = int(t[3], 16)
            lemmas = [t[4 + 2 * i].lower() for i in range(n)]
            rest = t[4 + 2 * n:]
            pc = int(rest[0])
            hyp = [rest[1 + 4 * i + 1] for i in range(pc) if rest[1 + 4 * i] in ("@", "@i")]
            self.syn[t[0]] = (lemmas, hyp, gloss.strip())
            self.hypo[t[0]] = [rest[1 + 4 * i + 1] for i in range(pc) if rest[1 + 4 * i] in ("~", "~i")]
        self.index = {}
        for line in (d / "index.noun").read_text(encoding="utf-8").splitlines():
            if line.startswith("  "):
                continue
            t = line.split()
            pc = int(t[3])
            self.index[t[0]] = t[6 + pc:]
        self.verbs = set()
        for line in (d / "index.verb").read_text(encoding="utf-8").splitlines():
            if not line.startswith("  "):
                self.verbs.add(line.split()[0])
        self._memo = {}

    def is_artifact(self, off: str) -> bool:
        if off in self._memo:
            return self._memo[off]
        self._memo[off] = False
        r = off == ARTIFACT_OFFSET or any(self.is_artifact(h) for h in self.syn[off][1])
        self._memo[off] = r
        return r

    def senses(self, term: str):
        return [(o, self.syn[o]) for o in self.index.get(term, [])]


# ---------------------------------------------------------------- main
def target_synsets(wn, co):
    """Synset(s) denoting the household-artifact sense of the object (empty if WordNet lacks it)."""
    if co in SENSE_OVERRIDE:
        return list(SENSE_OVERRIDE[co])
    for term in variants(co, OBJECT_VARIANTS):
        art = [o for o, _ in wn.senses(term) if wn.is_artifact(o)]
        if art:
            return art[:1]  # sense rank 1 among artifact senses (WordNet orders senses by frequency)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conceptnet-dump", required=True, type=Path)
    ap.add_argument("--wordnet-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--wordnet-zip-sha256", default="")
    args = ap.parse_args()

    obj_lookup = build_lookup(OBJECTS, OBJECT_VARIANTS)
    place_lookup = build_lookup(PLACES, PLACE_VARIANTS)
    objects = {o: {"places": {}, "synonyms": [], "rejected_synonyms": []} for o in OBJECTS}
    syn_seen = set()
    n_lines = 0
    wn = WordNet(args.wordnet_dir)
    vocab_terms = set(OBJECTS) | set(PLACES)
    targets = {co: target_synsets(wn, co) for co in OBJECTS}

    def judge_cn_synonym(canon, text, pos):
        """Return (rejection reason or None, wordnet corroboration True/False/None)."""
        if text == canon:
            return "same as the object name", None
        if pos not in POS_OK:
            return f"concept tagged POS '{pos}' (not a noun)", None
        if text in vocab_terms:
            return "another term of our own vocabulary (different concept or a room)", None
        if len(text.split("_")) > 3:
            return "long phrase (>3 words), likely a gloss, not a name", None
        sn = {o for o, _ in wn.senses(text)}
        if sn:
            if sn & set(targets[canon]):
                return None, True
            return "wrong sense: WordNet knows the word but shares no target synset with the object", False
        if text in wn.verbs:
            return "verb-only word (no noun entry in WordNet)", None
        return None, None  # not in WordNet: kept, flagged uncorroborated

    with gzip.open(args.conceptnet_dump, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            t = line.rstrip("\n").split("\t")
            if len(t) < 5 or t[1] not in (*EDGE_RELS, SYN_REL):
                continue
            if not (t[2].startswith("/c/en/") and t[3].startswith("/c/en/")):
                continue
            s, e = parse_concept(t[2]), parse_concept(t[3])
            if not s or not e:
                continue
            meta = json.loads(t[4])
            if t[1] in EDGE_RELS:
                for a, b, direction in ((s, e, "object->place"), (e, s, "place->object")):
                    # a is the object-side concept, b the place-side concept
                    if a[1] in obj_lookup and b[1] in place_lookup:
                        co, ov = obj_lookup[a[1]]
                        cp, pv = place_lookup[b[1]]
                        objects[co]["places"].setdefault(cp, []).append({
                            "relation": t[1].rsplit("/", 1)[1],
                            "direction": direction,
                            "weight": meta.get("weight"),
                            "surface": meta.get("surfaceText"),
                            "source": edge_source(meta),
                            "concept_uri": t[2] + " -> " + t[3],
                            "object_variant": ov,
                            "place_variant": pv,
                        })
            else:  # Synonym, symmetric: take either direction
                for a, b, uri in ((s, e, t[3]), (e, s, t[2])):
                    if a[1] in obj_lookup:
                        co, ov = obj_lookup[a[1]]
                        text = b[1]
                        key = (co, text, "conceptnet")
                        if key in syn_seen:
                            continue
                        syn_seen.add(key)
                        rec = {"text": text.replace("_", " "), "source": "conceptnet:" + edge_source(meta),
                               "weight": meta.get("weight"), "concept_uri": uri,
                               "object_variant": ov, "synsets": []}
                        why, corr = judge_cn_synonym(co, text, b[2])
                        rec["wordnet_corroborated"] = corr
                        rec["_cn_synsets"] = sorted({o for o, _ in wn.senses(text)})
                        if why:
                            rec["reason"] = why
                            objects[co]["rejected_synonyms"].append(rec)
                        else:
                            objects[co]["synonyms"].append(rec)

    # WordNet lemmas of the noun synsets of the object name and its variants
    for co in OBJECTS:
        for v in variants(co, OBJECT_VARIANTS):
            for off, (lemmas, _, gloss) in wn.senses(v):
                art = wn.is_artifact(off)
                for lem in lemmas:
                    if lem == co or lem in vocab_terms:
                        continue
                    key = (co, lem, "wordnet" + off)
                    if key in syn_seen:
                        continue
                    syn_seen.add(key)
                    rec = {"text": lem.replace("_", " "),
                           "source": f"wordnet-3.0:noun synset {off} via '{v}'",
                           "gloss": gloss[:120], "synsets": [off], "_cn_synsets": []}
                    if art:
                        objects[co]["synonyms"].append(rec)
                    else:
                        rec["reason"] = "wrong sense: synset is not under artifact.n.01"
                        objects[co]["rejected_synonyms"].append(rec)

    # lemmas of DIRECT hyponym synsets of the target synset
    for co in OBJECTS:
        for tsyn in targets[co]:
            for h in wn.hypo[tsyn]:
                for lem in wn.syn[h][0]:
                    if lem == co or lem in vocab_terms:
                        continue
                    key = (co, lem, "wordnet" + h)
                    if key in syn_seen:
                        continue
                    syn_seen.add(key)
                    objects[co]["synonyms"].append({
                        "text": lem.replace("_", " "),
                        "source": f"wordnet-3.0:noun synset {h} (direct hyponym of {tsyn})",
                        "gloss": wn.syn[h][2][:120], "synsets": [h], "_cn_synsets": []})

    def dominant(text_us, synset):
        sn = wn.index.get(text_us, [])
        return bool(sn) and sn[0] == synset

    # merge per text, then decide grammar_safe
    for co, o in objects.items():
        for key in ("synonyms", "rejected_synonyms"):
            merged = {}
            for r in o[key]:
                m = merged.setdefault(r["text"], dict(r))
                if r["source"] not in m["source"]:
                    m["source"] += " ; " + r["source"]
                m["synsets"] = sorted(set(m["synsets"]) | set(r["synsets"]))
                m["_cn_synsets"] = sorted(set(m["_cn_synsets"]) | set(r["_cn_synsets"]))
                for k in ("weight", "concept_uri", "object_variant", "wordnet_corroborated"):
                    if k in r and k not in m:
                        m[k] = r[k]
            o[key] = list(merged.values())
        tg = set(targets[co])
        hy = {h for t in tg for h in wn.hypo[t]}
        for r in o["synonyms"]:
            us = r["text"].replace(" ", "_")
            r["target_synset"] = sorted(tg)
            wn_hit = sorted(set(r["synsets"]) & (tg | hy))
            cn_hit = sorted(set(r["_cn_synsets"]) & (tg | hy)) if " " in r["text"] else []
            hit = wn_hit or cn_hit
            if not hit:
                r["grammar_safe"] = False
                r["reason"] = "not a lemma of the target household-artifact synset (nor a multiword ConceptNet term sharing it)"
            elif len(wn.index.get(us, [])) > 1 and not any(dominant(us, h) for h in hit):
                r["grammar_safe"] = False
                r["reason"] = "polysemous: target synset is not this word's dominant WordNet sense"
            else:
                r["grammar_safe"] = True
                r["matched_synset"] = hit[0]
                r["relation_to_target"] = "same_sense" if hit[0] in tg else "direct_hyponym"
        for r in o["rejected_synonyms"]:
            r["grammar_safe"] = False
        for lst in (o["synonyms"], o["rejected_synonyms"]):
            for r in lst:
                r.pop("_cn_synsets", None)
        for lst in o["places"].values():
            lst.sort(key=lambda x: -(x["weight"] or 0))
        o["synonyms"].sort(key=lambda x: x["text"])

    # ambiguity across objects (including collisions with other objects' own names)
    owners = {}
    for co, o in objects.items():
        for r in o["synonyms"]:
            if r["grammar_safe"]:
                owners.setdefault(r["text"], set()).add(co)
    own_names = {}
    for co in OBJECTS:
        for v in variants(co, OBJECT_VARIANTS):
            own_names.setdefault(v.replace("_", " "), set()).add(co)
    for co, o in objects.items():
        for r in o["synonyms"]:
            if not r["grammar_safe"]:
                continue
            others = (owners.get(r["text"], set()) | own_names.get(r["text"], set())) - {co}
            if others:
                r["grammar_safe"] = False
                r["reason"] = "ambiguous: also a safe synonym or name for " + ", ".join(sorted(others))
                r.pop("matched_synset", None)
                r.pop("relation_to_target", None)

    # variant_same_sense: edge reached through the object's own name or a grammar_safe synonym
    for co, o in objects.items():
        safe = {r["text"] for r in o["synonyms"] if r["grammar_safe"]}
        for lst in o["places"].values():
            for e in lst:
                v = e["object_variant"].replace("_", " ")
                e["variant_same_sense"] = e["object_variant"] == co or v in safe

    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    knowledge = {
        "version": 3,
        "generated_utc": generated,
        "sources": [
            {"name": "ConceptNet 5.7.0 assertions", "license": "CC BY-SA 4.0",
             "url": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
             "sha256": sha256_file(args.conceptnet_dump), "lines_scanned": n_lines},
            {"name": "WordNet 3.0", "license": "Princeton WordNet license (see LICENSE.txt)",
             "url": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
             "sha256": args.wordnet_zip_sha256},
        ],
        "place_vocabulary": PLACES,
        "target_synsets": {co: [{"synset": o, "lemmas": wn.syn[o][0], "gloss": wn.syn[o][2][:120]}
                                for o in targets[co]] for co in OBJECTS},
        "objects": objects,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "knowledge.json").write_text(
        json.dumps(knowledge, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # coverage report (plain text)
    rels = [r.rsplit("/", 1)[1] for r in EDGE_RELS]
    L = ["Coverage: objects x places. Cell = max edge weight over all relations/directions, '-' = no edge.",
         "Then per-relation edge counts (" + " ".join(rels) + ") and safe/all synonym counts.", ""]
    L.append("object | " + " | ".join(PLACES) + " | rel counts | syn_safe/syn")
    none_in_vocab = []
    tot = {r: 0 for r in rels}
    tot_dir = {"object->place": 0, "place->object": 0}
    for co in OBJECTS:
        pl = objects[co]["places"]
        cells = [f"{max(x['weight'] for x in pl[p]):.1f}" if p in pl else "-" for p in PLACES]
        cnt = {r: 0 for r in rels}
        for lst in pl.values():
            for x in lst:
                cnt[x["relation"]] += 1
                tot[x["relation"]] += 1
                tot_dir[x["direction"]] += 1
        syn = objects[co]["synonyms"]
        L.append(f"{co} | " + " | ".join(cells) + " | " + " ".join(str(cnt[r]) for r in rels)
                 + f" | {sum(r['grammar_safe'] for r in syn)}/{len(syn)}")
        if not pl:
            none_in_vocab.append(co)
    L.append("")
    L.append(f"Objects with >=1 place edge: {len(OBJECTS) - len(none_in_vocab)}/{len(OBJECTS)}")
    L.append("Objects with NO place evidence: " + (", ".join(none_in_vocab) or "none"))
    L.append("Edge totals by relation: " + ", ".join(f"{r}={tot[r]}" for r in rels))
    L.append("Edge totals by direction: " + ", ".join(f"{k}={v}" for k, v in tot_dir.items()))
    n_e = sum(len(l) for o in objects.values() for l in o["places"].values())
    n_s = sum(e["variant_same_sense"] for o in objects.values() for l in o["places"].values() for e in l)
    L.append(f"Edges with variant_same_sense=true: {n_s}/{n_e}")
    rt = {}
    for o in objects.values():
        for r in o["synonyms"]:
            if r["grammar_safe"]:
                rt[r["relation_to_target"]] = rt.get(r["relation_to_target"], 0) + 1
    L.append("grammar_safe synonyms by relation_to_target: " + str(rt))
    L.append(f"ConceptNet lines scanned: {n_lines}")
    (args.out_dir / "COVERAGE.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
