#!/usr/bin/env python3
"""
ndp — Neural Data Prefetch Architecture CLI

Commands:
  index [path]          Scan project directory and build the object index
  status                Show current tier assignments and object counts
  context               Print the currently staged context
  checkpoint [name]     Save a checkpoint of the current tier state
  restore <path>        Restore tier state from a checkpoint file
  bundle <name> [path]  Package current state into a portable .ndp bundle
  eval                  Score predictors on collected session logs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_index(args):
    from ndp.store import NDPStore
    from ndp.indexer import scan_project
    root = Path(args[0]).resolve() if args else Path.cwd()
    store = NDPStore()
    print(f"Indexing {root} ...")
    n = scan_project(root, store)
    stats = store.stats()
    print(f"Done — {n} files indexed, {stats['total']} total objects in store")


def cmd_status(args):
    from ndp.store import NDPStore
    store = NDPStore()
    stats = store.stats()
    print(f"Objects: {stats['total']}")
    for tier in ("hot", "warm", "cold", "evicted"):
        count = stats.get("by_tier", {}).get(tier, 0)
        if count:
            print(f"  {tier}: {count}")

    hot = store.get_by_tier("hot")
    if hot:
        print("\nHot tier:")
        for obj in hot:
            print(f"  [{obj.prior_usefulness:.2f}] {obj.source_path}")
    warm = store.get_by_tier("warm")
    if warm:
        print(f"\nWarm tier ({len(warm)} objects):")
        for obj in warm[:5]:
            print(f"  [{obj.prior_usefulness:.2f}] {obj.source_path}")
        if len(warm) > 5:
            print(f"  ... and {len(warm) - 5} more")


def cmd_context(args):
    ctx = Path.home() / ".ndp" / "context.md"
    if ctx.exists():
        print(ctx.read_text())
    else:
        print("No staged context. Open some files in Claude Code first.")


def cmd_checkpoint(args):
    from ndp.store import NDPStore
    from ndp import checkpoint
    store = NDPStore()
    name = args[0] if args else None
    path = checkpoint.save(store, name=name)
    print(f"Checkpoint saved: {path}")


def cmd_restore(args):
    if not args:
        print("Usage: ndp restore <checkpoint.json>")
        sys.exit(1)
    from ndp.store import NDPStore
    from ndp import checkpoint
    store = NDPStore()
    result = checkpoint.restore(store, Path(args[0]))
    print(f"Restored {result['restored']} objects from {result['checkpoint']}")


def cmd_bundle(args):
    if not args:
        print("Usage: ndp bundle <name> [output_dir]")
        sys.exit(1)
    from ndp.store import NDPStore
    from ndp.bundle import NDPBundle
    name = args[0]
    out_dir = Path(args[1]) if len(args) > 1 else Path.cwd()
    bundle_path = out_dir / f"{name}.ndp"
    store = NDPStore()
    bundle = NDPBundle(bundle_path).create(name, cwd=str(Path.cwd()))
    bundle.export_objects(store)
    bundle.pack_sessions()
    print(f"Bundle created: {bundle_path}")
    print(f"  manifest: {bundle.manifest()}")


def cmd_eval(args):
    from eval.harness import run_eval
    run_eval()


COMMANDS = {
    "index": cmd_index,
    "status": cmd_status,
    "context": cmd_context,
    "checkpoint": cmd_checkpoint,
    "restore": cmd_restore,
    "bundle": cmd_bundle,
    "eval": cmd_eval,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
