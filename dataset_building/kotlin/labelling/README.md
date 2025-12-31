## Static Analysis (Semgrep OWASP rules)

We scan each Kotlin snippet with a custom OWASP-inspired Semgrep ruleset and store findings as JSON.

```bash
semgrep \
  --config kotlin_owasp_rules.yml \
  --max-target-bytes 5000000 \
  --json -o semgrep_kotlin_owasp.json \
  --verbose \
  kotlin_owasp_functions 2> semgrep_owasp_verbose.log
