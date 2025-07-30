Here’s a complete example of how to integrate SyntaxLab’s mutation testing and validation engine into a CI pipeline — both for GitHub Actions and GitLab CI/CD.

⸻

✅ GitHub Actions: SyntaxLab Mutation Validation

📄 .github/workflows/syntaxlab-validation.yml

name: SyntaxLab Mutation Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  syntaxlab-validate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Install CLI (binary or NPM)
        run: |
          curl -sSL https://get.syntaxlab.ai/install.sh | bash
          syntaxlab --version

      - name: Run Mutation Testing
        run: |
          syntaxlab validate \
            --mutation \
            --threshold 0.9 \
            --output-format json \
            --report-path ./reports/mutation.json

      - name: Upload Mutation Report
        uses: actions/upload-artifact@v3
        with:
          name: mutation-report
          path: ./reports/mutation.json

      - name: Fail if Mutation Score Too Low
        run: |
          score=$(jq .mutation_score ./reports/mutation.json)
          echo "Mutation Score: $score"
          if (( $(echo "$score < 0.9" | bc -l) )); then
            echo "❌ Mutation score below threshold"
            exit 1
          fi


⸻

🛠️ GitLab CI/CD: SyntaxLab Validator Integration

📄 .gitlab-ci.yml

stages:
  - test
  - validate

syntaxlab-validation:
  stage: validate
  image: node:20
  before_script:
    - curl -sSL https://get.syntaxlab.ai/install.sh | bash
    - syntaxlab --version
  script:
    - mkdir -p reports
    - syntaxlab validate \
        --mutation \
        --threshold 0.9 \
        --output-format json \
        --report-path reports/mutation.json
    - |
      score=$(jq .mutation_score reports/mutation.json)
      echo "🔍 Mutation Score: $score"
      if [ "$(echo "$score < 0.9" | bc)" -eq 1 ]; then
        echo "❌ Mutation score below threshold"
        exit 1
      fi
  artifacts:
    paths:
      - reports/mutation.json
    expire_in: 1 week


⸻

📦 Output Artifact

Example mutation.json artifact:

{
  "file_count": 4,
  "mutants_tested": 92,
  "mutants_killed": 86,
  "mutation_score": 0.9348,
  "survivors": [
    {
      "id": "M005",
      "file": "stack.py",
      "line": 22,
      "description": "Modified return value to constant 0"
    }
  ]
}


⸻

🧠 Bonus: Slack Notification on Score Failure

You can hook into GitHub/GitLab’s webhook system or use a post-step like this:

if (( $(echo "$score < 0.9" | bc -l) )); then
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"🧪 Mutation score too low: $score\"}" \
    https://hooks.slack.com/services/YOUR/SLACK/URL
fi
