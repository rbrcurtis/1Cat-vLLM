# SPDX-License-Identifier: Apache-2.0
"""KV archive reachability support.

The tiered KV offload archive is content-addressed and append-only, so a
prompt that rewrites its own middle (compaction) forks the block-hash chain:
every block after the divergence point is a new hash, and the previous branch
becomes permanently unreachable while still occupying disk.

To garbage-collect by *reachability* instead of age, the engine needs to know
which chain belongs to which session, because only a session's newest chain
can ever be matched again. Requests are tagged by the client (pi) with

    X-Request-Id: pi-<session id>-<per-process counter>

vLLM already honours ``X-Request-Id`` as the engine request id, so the tag
arrives as ``request_id`` with no extra plumbing and is visible to the
offloading connector as ``ReqContext.req_id``.

Two hooks live here:

* ``archive_session_for`` — used by the tiering manager to suppress archive
  writes for untagged traffic, so only pi blocks ever reach disk.
* ``log_chain`` — appends ``{ts, request_id, tokens, hashes}`` to a JSONL file
  consumed by the GC, which pins the newest chain per session and deletes
  everything else before considering age-based eviction.

Both are inert unless their environment variables are set.
"""

import json
import re
import time
from typing import Sequence

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_logged_path = False

# ``<session>-<counter>``: session ids may contain dashes, the counter is the
# first all-digits segment.
_SESSION_RE = re.compile(r"([A-Za-z0-9_.:-]+?)-\d+(?:-|$)")


def archive_tag_prefix() -> str:
    """Request-id prefix that marks traffic as archivable (default ``pi-``)."""
    return envs.VLLM_KV_ARCHIVE_TAG_PREFIX


def chain_log_path() -> str:
    global _logged_path
    path = envs.VLLM_KV_ARCHIVE_CHAIN_LOG
    if not _logged_path:
        _logged_path = True
        logger.info(
            "KV archive reachability: tag prefix %r, chain log %s",
            envs.VLLM_KV_ARCHIVE_TAG_PREFIX,
            path or "(disabled)",
        )
    return path


def archive_session_for(req_id: str | None) -> str | None:
    """Return the session id if this request may be archived, else None.

    The engine request id is not the raw header: the OpenAI frontend wraps it
    as ``chatcmpl-pi-<session>-<counter>-<suffix>`` (request-id prefix, then the
    caller's header value, then the server's own uniqueness suffix). So the tag
    is located anywhere in the id, and the first ``-<digits>`` segment after it
    ends the session (a session id may itself contain dashes).
    """
    prefix = archive_tag_prefix()
    if not req_id or not prefix:
        return None
    idx = req_id.find(prefix)
    if idx < 0:
        return None
    matches = _SESSION_RE.match(req_id, idx + len(prefix))
    return matches.group(1) if matches else None


def log_chain(req_id: str, num_prompt_tokens: int, hashes: Sequence[bytes]) -> None:
    """Record one request's prompt block-hash chain for reachability GC."""
    log_path = chain_log_path()
    if not log_path:
        return
    session = archive_session_for(req_id)
    if session is None or not hashes:
        return
    logger.info(
        "KV archive chain: session=%s tokens=%d blocks=%d",
        session,
        num_prompt_tokens,
        len(hashes),
    )
    record = {
        "ts": time.time(),
        "request_id": req_id,
        "session": session,
        "tokens": num_prompt_tokens,
        "hashes": [h.hex() if isinstance(h, bytes) else str(h) for h in hashes],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        # Never let bookkeeping break serving.
        pass
