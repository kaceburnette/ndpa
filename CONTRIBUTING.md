# Contributing to NDPA

NDPA is early. Contributions are welcome, especially on:

- **Eval datasets** — more real session logs sharpen the prediction kernel
- **New predictors** — try a different feature set, beat the heuristic
- **Platform SDKs** — Rust, Go, Swift, Kotlin, Ruby
- **Integrations** — wire NDPA into an existing AI agent / IDE / app
- **Bug reports and design feedback** — issues are encouraged

## Dev setup

```bash
git clone https://github.com/kaceburnette/ndp
cd ndp
pip install -r requirements.txt
python3 -m ndp.config        # one-time setup
```

## Running the eval

```bash
python3 -m eval.harness            # file-level
python3 -m eval.conversation_eval  # conversation-level
```

## Building the SDKs

```bash
cd sdk/python      && pip install -e .
cd sdk/typescript  && npm install && npm run build
```

## Code style

- Python: stdlib only where reasonable, no over-engineering, no premature abstractions
- TypeScript: zero runtime dependencies in the SDK
- Match what's already here. Small focused PRs.

## License

By contributing you agree your contribution is licensed under the MIT License.
