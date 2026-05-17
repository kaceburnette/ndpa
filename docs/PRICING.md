# NDPA Preview Pricing

This is preview pricing for planning and early customer conversations. It is
not a committed public price list.

NDPA is priced by layer because the layers have different cost drivers:
Predict and Memory are lightweight metadata operations, Hydration moves raw
bytes, and Reasoning calls an LLM.

## Preview Unit Prices

| Layer | What is billed | Preview price | Cost driver |
|---|---:|---:|---|
| Predict | 1,000 staged prediction requests | $0.03 / 1K | CPU and compact metadata reads |
| Memory | 1,000 explicit retrieval requests | $0.10 / 1K | CPU, compact metadata search, ranking |
| Hydration | 1,000 raw context fetches | $0.05 / 1K fetches + $0.15 / GB returned + $0.03 / GB-month stored | Blob/object/local storage reads and bytes returned |
| Reasoning BYOK | 1,000 answer-orchestration calls | $5 / 1K + customer pays model provider | Retrieval, hydration, prompt assembly, routing, observability |
| Reasoning Hosted | 1,000 answer-generation calls | $40 / 1K standard budget | LLM input/output token cost plus orchestration |

## Unit Economics

Predict and Memory are the core NDPA lanes. They do not require an LLM,
embeddings, vector search, or a graph extraction model in the hot path. The
hosted core stores compact metadata such as handles, previews, scores, top
terms, timestamps, storage keys, and byte counts.

Hydration is separate because it fetches raw context. A 1 KB context preview
and a 1 MB transcript have very different storage and bandwidth costs even if
they count as one fetch. Hydration pricing should therefore include a per-fetch
meter plus storage and bandwidth accounting.

Reasoning is also separate. If NDPA generates an answer over retrieved or
hydrated context, the LLM provider cost is the main variable cost. The current
full 500-row LongMemEval-S repaired Reasoning run scored **85.2%** with
`gpt-5-mini` generation and `gpt-4.1-mini` LLM judging. The measured generation
and judge cost for that hard benchmark profile was about **$7.48 / 500 rows**,
or **$14.96 / 1K Reasoning calls**.

That measured benchmark cost is why hosted Reasoning cannot be priced like
Predict or Memory. At $40 / 1K hosted Reasoning calls, the measured hard-profile
gross margin is about 63% before hosting, support, failed retries, and payment
fees. At $30 / 1K, the margin drops to about 50%. BYOK exists for customers who
want NDPA's orchestration but want model spend billed directly to their own LLM
provider account.

Large-context, high-output, or regulated deployments should be quoted with
explicit token budgets, pass-through overages, or dedicated infrastructure.

## Example Monthly Usage

| Usage | Formula | Preview cost |
|---|---:|---:|
| 1M Predict requests | 1,000 x $0.03 | $30 |
| 1M Memory requests | 1,000 x $0.10 | $100 |
| 100K Hydration fetches | 100 x $0.05 | $5 plus returned bytes and stored bytes |
| 100K Reasoning BYOK calls | 100 x $5 | $500 plus the customer's model bill |
| 100K Reasoning Hosted calls | 100 x $40 | $4,000; measured hard-profile model COGS would be about $1,496 |

These examples are arithmetic examples only. They do not include committed-use
discounts, enterprise minimums, support, custom retention, dedicated
infrastructure, or regulated deployment requirements.

## Self-Serve Packaging

| Plan | Included preview allowance | Pricing after allowance |
|---|---:|---|
| Free | 50K Predict, 10K Memory, 1K Hydration fetches, 100 MB stored, 25 hosted Reasoning calls | Upgrade required after allowance |
| Pay-as-you-go | No monthly platform commitment during preview | Unit prices above |
| Scale / Enterprise | Custom retention, dedicated infra, SSO, higher rate limits | Quote by volume and deployment model |

A small monthly minimum invoice may be added after preview so tiny paid
workspaces do not get eaten by payment processing and support overhead.

## Packaging Guidance

For self-serve customers, expose Predict and Memory as the default product.
Hydration should be available when applications need exact raw context, but it
should remain metered by bytes so large transcripts do not distort core margins.

For enterprise customers, quote separate lines for:

- Core API usage: Predict and Memory requests.
- Raw context: Hydration fetches, stored bytes, retained days, and egress.
- Reasoning: BYOK vs hosted, model choice, token budget, context cap, retry
  policy, and whether NDPA or the customer owns the LLM provider account.
- Deployment: shared hosted, dedicated hosted, self-hosted, or embedded.

## Should We Push Reasoning Past 90% Before Launch?

No, not before the launch surface is wired. The current full-run Reasoning
score is **85.2%**, and getting to 90% requires roughly 24 additional correct
rows out of 500. That is likely a focused product/research pass, not a pricing
or packaging blocker.

Launch with 85.2% as the honest current best. Keep Reasoning labeled as the
premium optional layer, then improve the weak categories after customers can
actually sign up and use the cheaper Predict/Memory/Hydration layers.

## Hydration Microbench

The local benchmark reads 1 KB, 10 KB, 100 KB, and 1 MB blobs through
`hydrate_local_blob`. It is a local filesystem benchmark only. It is not a
hosted API latency claim and it is not object-storage latency.

Run:

```bash
python3 -m eval.hydration_microbench
```

Latest local run is stored in `eval/hydration_microbench_results.json`.

Latest local result, 300 measured reads after 20 warmups:

| Blob size | P50 | P95 |
|---:|---:|---:|
| 1 KB | 0.0561 ms | 0.0625 ms |
| 10 KB | 0.0563 ms | 0.0657 ms |
| 100 KB | 0.0695 ms | 0.0789 ms |
| 1 MB | 0.1782 ms | 0.2073 ms |

Do not claim low-latency hosted Hydration until the hosted API and object
storage path have been measured directly.
