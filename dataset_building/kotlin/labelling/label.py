import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


base_dir = Path(__file__).resolve().parent

input_csv = base_dir / "kotlin_security_classes_with_ids.csv"
semgrep_json = base_dir / "semgrep_kotlin_owasp.json"
output_csv = base_dir / "kotlin_security_classes_owasp_labeled.csv"


def load_results(p: Path) -> list:
    with p.open("r", encoding="utf-8") as f:
        return (json.load(f) or {}).get("results", [])


def get_sample_id(r: dict) -> str:
    path = r.get("path") or (r.get("extra") or {}).get("path") or ""
    return Path(path).stem if path else ""


def aggregate(results: list) -> pd.DataFrame:
    agg = defaultdict(lambda: {"rule_ids": set(), "severities": [], "cwes": set(), "owasps": set()})

    for r in results:
        sid = get_sample_id(r)
        if not sid:
            continue

        extra = r.get("extra") or {}
        rule_id = extra.get("rule_id") or r.get("check_id") or ""
        severity = (extra.get("severity") or "").upper()

        meta = extra.get("metadata") or {}
        cwe_val = meta.get("cwe") or meta.get("CWE")
        owasp_val = meta.get("owasp") or meta.get("OWASP")

        e = agg[sid]
        if rule_id:
            e["rule_ids"].add(str(rule_id))
        if severity:
            e["severities"].append(severity)

        if isinstance(cwe_val, list):
            for x in cwe_val:
                e["cwes"].add(str(x))
        elif cwe_val:
            e["cwes"].add(str(cwe_val))

        if isinstance(owasp_val, list):
            for x in owasp_val:
                e["owasps"].add(str(x))
        elif owasp_val:
            e["owasps"].add(str(owasp_val))

    rows = []
    for sid, info in agg.items():
        sev = info["severities"]
        high = sum(s == "ERROR" for s in sev)
        med = sum(s == "WARNING" for s in sev)
        low = sum(s == "INFO" for s in sev)
        total = len(sev)

        rows.append({
            "sample_id": sid,
            "owasp_rule_count": total,
            "owasp_high_count": high,
            "owasp_medium_count": med,
            "owasp_low_count": low,
            "owasp_rule_ids": ";".join(sorted(info["rule_ids"])) if info["rule_ids"] else "",
            "owasp_cwe_list": ";".join(sorted(info["cwes"])) if info["cwes"] else "",
            "owasp_owasp_list": ";".join(sorted(info["owasps"])) if info["owasps"] else "",
            "owasp_hp_vuln": int(high > 0),
            "owasp_hr_vuln": int(total > 0),
        })

    return pd.DataFrame(rows)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = [
        "owasp_rule_count", "owasp_high_count", "owasp_medium_count", "owasp_low_count",
        "owasp_hp_vuln", "owasp_hr_vuln",
    ]
    str_cols = ["owasp_rule_ids", "owasp_cwe_list", "owasp_owasp_list"]

    for c in int_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)

    for c in str_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    return df


def main():
    results = load_results(semgrep_json)
    df_sem = aggregate(results)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    out = df.merge(df_sem, on="sample_id", how="left")
    out = coerce(out)

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} hp_pos={int(out['owasp_hp_vuln'].sum())} hr_pos={int(out['owasp_hr_vuln'].sum())} -> {output_csv}")


if __name__ == "__main__":
    main()
