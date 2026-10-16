# Security

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
reporting instead: **Security → Report a vulnerability** on
https://github.com/700799/tunepac. You will get an acknowledgement within a
few days.

## What to know before you train

- **`base_model.trust_remote_code` defaults to `false`.** Setting it to `true`
  executes Python shipped inside the model repository on the Hugging Face Hub
  with your user's permissions. Only enable it for a model you have read the
  code of, pinned by `base_model.revision`.
- **Configs are data, not code.** The YAML loader uses `yaml.safe_load`, and
  the schema rejects unknown keys, so a config file cannot inject behavior.
  Datasets are likewise only ever parsed, never executed.
- **Dependencies.** Training pulls in torch, transformers, TRL, PEFT and
  bitsandbytes. Dependabot opens pull requests for updates to the pinned GitHub
  Actions and to pip dependencies; review them like any other change.
