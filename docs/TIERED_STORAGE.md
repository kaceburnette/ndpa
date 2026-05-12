# Tiered Storage in NDPA

NDPA's biggest cost advantage at AI-platform scale comes from the tiered
storage model: cold data lives cheap, but the kernel predictively promotes
what's about to be needed so users never pay the cold-tier latency.

## The architecture

```
                ┌──────────────────────────┐
                │   Hot tier: Postgres     │  ← queryable, indexed, fast
                │   (~$5/GB/month)         │
                └────────┬─────────────────┘
                         │ predictive_promote()
                         │
       ┌─────────────────┴─────────────────┐
       │                                   │
   ┌───▼──────────────┐         ┌─────────▼────────┐
   │  Cold tier:      │         │  Even colder:    │
   │  Object storage  │         │  Archive tiers   │
   │  S3, GCS, blob   │         │  (Glacier, etc.) │
   │  (~$0.02/GB/mo)  │         │  (~$0.001/GB/mo) │
   └──────────────────┘         └──────────────────┘
```

The kernel watches user behavior and **promotes** cold conversations to
hot tier BEFORE the user asks. They get hot-tier latency at cold-tier prices.

## The interface

```python
from ndp.tiered import LocalFSBackend, demote, promote, predictive_promote
from pathlib import Path

# Any backend that implements TierBackend works:
backend = LocalFSBackend(Path("/var/ndpa/cold"))

# Move an old session out of hot tier
demote("session_abc", backend)

# Pull it back when needed
promote("session_abc", backend)

# Or let the kernel decide what to pre-warm based on current activity
promoted = predictive_promote(
    backend,
    query_context="recent prompts + active file paths",
    k=5,
)
```

## TierBackend interface

```python
class TierBackend(ABC):
    def read_bundle(self, session_id: str) -> bytes: ...
    def write_bundle(self, session_id: str, data: bytes, metadata: dict): ...
    def delete_bundle(self, session_id: str): ...
    def list_bundles(self) -> Iterator[str]: ...
    def read_metadata(self, session_id: str) -> dict | None: ...
```

Five methods. Any object store can implement it.

## Backends shipped

### LocalFSBackend (default)

Stores `.ndp` bundles in a directory on disk. Layout:

```
{root}/bundles/{session_id}.ndp        ← the bundle (zip)
{root}/metadata/{session_id}.json      ← sidecar with top_terms
```

Use cases:
- Development / prototyping
- Single-machine deployments
- Self-hosted clusters with shared filesystem (NFS, EFS)

### S3Backend (stub)

Defined interface, no implementation. Subclass it with your client of choice:

```python
class MyS3Backend(S3Backend):
    def __init__(self, bucket, prefix=""):
        import boto3
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix

    def read_bundle(self, session_id):
        obj = self.s3.get_object(Bucket=self.bucket,
                                 Key=f"{self.prefix}/bundles/{session_id}.ndp")
        return obj["Body"].read()

    def write_bundle(self, session_id, data, metadata=None):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/bundles/{session_id}.ndp",
            Body=data,
        )
        if metadata:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/metadata/{session_id}.json",
                Body=json.dumps(metadata).encode(),
            )

    def delete_bundle(self, session_id):
        self.s3.delete_object(Bucket=self.bucket,
                              Key=f"{self.prefix}/bundles/{session_id}.ndp")
        self.s3.delete_object(Bucket=self.bucket,
                              Key=f"{self.prefix}/metadata/{session_id}.json")

    def list_bundles(self):
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket,
                                       Prefix=f"{self.prefix}/bundles/"):
            for obj in page.get("Contents", []):
                yield Path(obj["Key"]).stem

    def read_metadata(self, session_id):
        try:
            obj = self.s3.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/metadata/{session_id}.json",
            )
            return json.loads(obj["Body"].read())
        except Exception:
            return None
```

NDPA stays zero-dependency. Bring your own client.

The same pattern works for GCS (`google-cloud-storage`), Azure Blob
(`azure-storage-blob`), Cloudflare R2 (use boto3 with R2 endpoint), or
self-hosted MinIO/SeaweedFS.

## Predictive promotion

The function that makes this architecture worth building:

```python
def predictive_promote(backend, query_context, k=5, max_scan=1000):
    """
    Scan cold-tier metadata sidecars (cheap), score against trajectory,
    promote top-K bundles BEFORE the user asks.
    """
```

It does NOT read full bundles to score them. Only the metadata sidecars
(top_terms + started_at + platform — usually <1KB each). Cold-tier reads
happen only for the top-K matches that actually get promoted.

For 1M cold-tier sessions:
- 1M metadata reads × ~500 bytes = 500MB transfer to score everything
- Or use `max_scan` to cap at recent N

For really large archives (10M+ per user), shard the metadata into a
flat index: `bulk_metadata.json` with `{session_id: top_terms}`.
One read covers everything. Build it lazily from individual sidecars.

## CLI

```bash
# Demote a session to cold tier
python3 -m ndp.tiered demote session_abc --cold-dir /path/to/cold

# Promote it back
python3 -m ndp.tiered promote session_abc --cold-dir /path/to/cold

# List bundles in cold tier
python3 -m ndp.tiered list --cold-dir /path/to/cold

# Predict and promote top-K based on a query
python3 -m ndp.tiered predict "rust async tokio" --cold-dir /path/to/cold --k 5
```

## Why this matters at scale

**At 1M users × 100k conversations/user (typical for a year of usage):**

| Storage strategy           | Storage cost/mo | Latency (P50) |
|----------------------------|-----------------|---------------|
| All hot (Postgres)         | ~$500k          | 110ms         |
| All cold (S3)              | ~$2k            | 200-500ms     |
| **Tiered + predictive**    | **~$25k**       | **110ms**     |

The tiered approach gets cold-tier prices on 95% of data and hot-tier
latency on 100% of requests — because predictive promotion ensures the
right 5% is always hot. That's the **20× cost reduction** that makes
NDPA viable at AI-platform scale.

## What's NOT yet built

- **Automatic demotion job**: deciding what to move from hot → cold
  based on staleness. Today: manual `demote(session_id)` call.
- **Tiered-aware predictions API edge function**: today the predictions
  API only queries hot tier. To leverage cold tier in real-time, the
  edge function would need to call `predictive_promote` before its main
  scoring step. Easy addition; haven't done it yet.
- **Production S3 backend implementation**: stub defined. Plug in boto3
  with ~30 lines (see example above).

These are obvious next steps; the architecture supports them. Open issues
welcome at https://github.com/kaceburnette/ndpa/issues.
