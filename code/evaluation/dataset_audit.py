#!/usr/bin/env python3
"""Audit CSV contracts, image references, joins, and image signatures (stdlib only)."""
from __future__ import annotations
import collections, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]
OUTPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object", "evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]
ISSUES = {"dent","scratch","crack","glass_shatter","broken_part","missing_part","torn_packaging","crushed_packaging","water_damage","stain","none","unknown"}
STATUSES = {"supported","contradicted","not_enough_information"}
SEVERITIES = {"none","low","medium","high","unknown"}
FLAGS = {"none","blurry_image","cropped_or_obstructed","low_light_or_glare","wrong_angle","wrong_object","wrong_object_part","damage_not_visible","claim_mismatch","possible_manipulation","non_original_image","text_instruction_present","user_history_risk","manual_review_required"}
PARTS = {
 "car":{"front_bumper","rear_bumper","door","hood","windshield","side_mirror","headlight","taillight","fender","quarter_panel","body","unknown"},
 "laptop":{"screen","keyboard","trackpad","hinge","lid","corner","port","base","body","unknown"},
 "package":{"box","package_corner","package_side","seal","label","contents","item","unknown"},
}
def read_csv(path):
 with path.open(encoding="utf-8-sig", newline="") as f:
  reader=csv.DictReader(f); return list(reader.fieldnames or []),list(reader)
def detect_format(path):
 try: h=path.read_bytes()[:64]
 except OSError: return "unreadable"
 if h.startswith(b"\xff\xd8\xff"): return "jpeg"
 if h.startswith(b"\x89PNG\r\n\x1a\n"): return "png"
 if h.startswith((b"GIF87a",b"GIF89a")): return "gif"
 if len(h)>=12 and h[:4]==b"RIFF" and h[8:12]==b"WEBP": return "webp"
 if len(h)>=12 and h[4:8]==b"ftyp":
  if b"avif" in h[8:32] or b"avis" in h[8:32]: return "avif"
  if any(x in h[8:32] for x in (b"heic",b"heix",b"mif1")): return "heif_or_heic"
 if h.startswith(b"BM"): return "bmp"
 if h.startswith((b"II*\x00",b"MM\x00*")): return "tiff"
 return "unknown"
def count(rows,key): return dict(sorted(collections.Counter(r.get(key,"") for r in rows).items()))
def audit():
 errors=[]; warnings=[]
 required=[DATA/"claims.csv",DATA/"sample_claims.csv",DATA/"user_history.csv",DATA/"evidence_requirements.csv",ROOT/"code/main.py",ROOT/"code/evaluation/main.py"]
 missing_required=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
 if missing_required: errors.append("Missing required files: "+", ".join(missing_required))
 try:
  ih,claims=read_csv(DATA/"claims.csv"); sh,samples=read_csv(DATA/"sample_claims.csv")
  hh,history=read_csv(DATA/"user_history.csv"); rh,requirements=read_csv(DATA/"evidence_requirements.csv")
  oh,_=read_csv(DATA/"output.csv") if (DATA/"output.csv").exists() else ([],[])
 except (OSError,csv.Error) as e:
  return {"errors":errors+[f"CSV read error: {e}"],"warnings":warnings}
 if ih!=INPUT_COLUMNS: errors.append(f"claims.csv schema mismatch: {ih}")
 if sh!=OUTPUT_COLUMNS: errors.append("sample_claims.csv does not match the required 14-column output schema")
 if oh and oh!=OUTPUT_COLUMNS: errors.append("output.csv header does not match the required output schema")
 if not {"user_id","past_claim_count","history_flags","history_summary"}.issubset(hh): errors.append("user_history.csv missing required columns")
 if not {"requirement_id","claim_object","applies_to","minimum_image_evidence"}.issubset(rh): errors.append("evidence_requirements.csv missing required columns")
 allowed_objects={"car","laptop","package"}
 unknown_objects=sorted({r.get("claim_object","") for r in claims+samples}-allowed_objects)
 if unknown_objects: errors.append(f"Unexpected claim_object values: {unknown_objects}")
 history_ids={r.get("user_id","") for r in history}
 missing_history=[]; missing_images=[]; refs=[]
 for split,rows in (("test",claims),("sample",samples)):
  for n,row in enumerate(rows,start=2):
   if row.get("user_id","") not in history_ids: missing_history.append(f"{split} row {n}: {row.get('user_id','')}")
   for raw in filter(None,(x.strip() for x in row.get("image_paths","").split(";"))):
    p=DATA/raw; refs.append(p)
    if not p.is_file(): missing_images.append({"split":split,"row":n,"path":raw})
 if missing_images: errors.append(f"{len(missing_images)} referenced image files are missing")
 if missing_history: warnings.append(f"{len(missing_history)} rows have no user history record")
 unique=sorted(set(refs)); formats=collections.Counter(); mismatches=[]
 for p in unique:
  fmt=detect_format(p); formats[fmt]+=1; ext=p.suffix.lower().lstrip(".")
  if fmt not in {"unknown","unreadable"} and not ((ext in {"jpg","jpeg"} and fmt=="jpeg") or ext==fmt):
   mismatches.append({"path":str(p.relative_to(ROOT)),"extension":ext,"detected_format":fmt})
 if mismatches: warnings.append(f"{len(mismatches)} image extensions do not match detected file formats")
 invalid=[]
 for n,r in enumerate(samples,start=2):
  fields=[("claim_status",r.get("claim_status",""),STATUSES),("issue_type",r.get("issue_type",""),ISSUES),("severity",r.get("severity",""),SEVERITIES)]
  fields.append(("object_part",r.get("object_part",""),PARTS.get(r.get("claim_object",""),set())))
  for key,value,allowed in fields:
   if value not in allowed: invalid.append({"row":n,"field":key,"value":value})
  flags=[x.strip() for x in r.get("risk_flags","").split(";") if x.strip()]
  if not flags or any(x not in FLAGS for x in flags): invalid.append({"row":n,"field":"risk_flags","value":r.get("risk_flags","")})
  for key in ("evidence_standard_met","valid_image"):
   if r.get(key,"").lower() not in {"true","false"}: invalid.append({"row":n,"field":key,"value":r.get(key,"")})
 if invalid: warnings.append(f"{len(invalid)} sample label fields violate allowed-value rules")
 return {
  "repository_root":str(ROOT),"errors":errors,"warnings":warnings,
  "counts":{"test_claim_rows":len(claims),"sample_claim_rows":len(samples),"history_rows":len(history),"evidence_requirement_rows":len(requirements),
   "test_image_references":sum(bool(r.get("image_paths","").strip()) and len(r["image_paths"].split(";")) for r in claims),
   "sample_image_references":sum(bool(r.get("image_paths","").strip()) and len(r["image_paths"].split(";")) for r in samples),
   "unique_referenced_images":len(unique),"detected_image_formats":dict(sorted(formats.items())),
   "missing_image_references":len(missing_images),"image_extension_mismatches":len(mismatches)},
  "test_claims_by_object":count(claims,"claim_object"),"sample_claims_by_object":count(samples,"claim_object"),
  "sample_label_distributions":{k:count(samples,k) for k in ("claim_status","issue_type","evidence_standard_met","valid_image","severity")},
  "missing_images":missing_images,"missing_history_rows":missing_history,"image_extension_mismatches":mismatches,"invalid_sample_fields":invalid
 }
def main():
 report=audit(); print(json.dumps(report,indent=2,ensure_ascii=False))
 c=report.get("counts",{})
 print(f"AUDIT SUMMARY: {c.get('test_claim_rows',0)} test claims; {c.get('sample_claim_rows',0)} samples; {c.get('missing_image_references','?')} missing images; {c.get('image_extension_mismatches','?')} extension mismatches; {len(report.get('errors',[]))} errors; {len(report.get('warnings',[]))} warnings.",file=sys.stderr)
 return 1 if report.get("errors") else 0
if __name__=="__main__": raise SystemExit(main())
