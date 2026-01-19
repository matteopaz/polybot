from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import PolymarketSDK

EVENTS_PATH = Path("data/events.json")
OUTPUT_DIR = Path("data/trades")
MIN_VALUE_USD = 500.0
PAGE_LIMIT = 10000
MAX_OFFSET = 10000
SLEEP_SECONDS = 0.0


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def market_volume(market_raw: dict[str, Any]) -> float:
    return parse_float(
        market_raw.get("volume")
        or market_raw.get("volumeNum")
        or market_raw.get("volumeClob")
        or 0
    )


def market_condition_id(market_raw: dict[str, Any]) -> str | None:
    cid = market_raw.get("conditionId") or market_raw.get("condition_id")
    if isinstance(cid, str) and cid.startswith("0x") and len(cid) == 66:
        return cid
    return None


def event_metadata(event) -> dict[str, Any]:
    raw = event.raw if isinstance(event.raw, dict) else {}
    created_at = raw.get("createdAt") or raw.get("creationDate")
    if created_at is None and event.created_at:
        created_at = event.created_at.isoformat()
    resolved_at = raw.get("resolvedAt") or raw.get("resolutionDate") or raw.get("endDate")
    if resolved_at is None and event.end_date:
        resolved_at = event.end_date.isoformat()
    resolved_flag = raw.get("resolved")
    if resolved_flag is None:
        resolved_flag = raw.get("closed")
    resolution = "yes" if resolved_flag else "no"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "created_at": created_at,
        "resolved_at": resolved_at,
        "resolution": resolution,
    }


def fetch_trades_for_market(
    client: PolymarketSDK, condition_id: str
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    seen: set[str] = set()

    for side in ("BUY", "SELL"):
        offset = 0
        while offset <= MAX_OFFSET:
            page = client.get_trades(
                market=[condition_id],
                limit=PAGE_LIMIT,
                offset=offset,
                taker_only=False,
                filter_type="CASH",
                filter_amount=MIN_VALUE_USD,
                side=side,
            )
            if not page:
                break

            for trade in page:
                if trade.price is None or trade.size is None:
                    continue
                value = trade.price * trade.size
                if value < MIN_VALUE_USD:
                    continue
                account = trade.proxy_wallet
                if not account:
                    continue
                key = (
                    f"{trade.transaction_hash}-{trade.timestamp}-{account}-"
                    f"{trade.condition_id}-{trade.side}-{trade.size}-{trade.price}"
                )
                if key in seen:
                    continue
                seen.add(key)
                trades.append(
                    {
                        "account": account,
                        "side": trade.outcome or trade.side,
                        "value": value,
                        "timestamp": trade.timestamp.isoformat()
                        if trade.timestamp
                        else None,
                    }
                )

            if len(page) < PAGE_LIMIT:
                break
            if offset + PAGE_LIMIT > MAX_OFFSET:
                break
            offset += PAGE_LIMIT

    return trades


def main() -> None:
    client = PolymarketSDK()
    events = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))

    event_cache = {}
    markets: list[dict[str, Any]] = []

    for entry in events:
        event_id = entry.get("id")
        if event_id is None:
            continue
        event = client.get_event_by_id(str(event_id), include_markets=True)
        if not event or not event.markets:
            continue
        event_cache[event.id] = event
        for market in event.markets:
            if not isinstance(market.raw, dict):
                continue
            condition_id = market_condition_id(market.raw)
            if not condition_id:
                continue
            markets.append(
                {
                    "event_id": event.id,
                    "market_id": market.id,
                    "condition_id": condition_id,
                    "volume": market_volume(market.raw),
                }
            )

    markets.sort(key=lambda item: item["volume"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, market in enumerate(markets, start=1):
        event = event_cache.get(market["event_id"])
        if event is None:
            continue
        event_dir = OUTPUT_DIR / str(event.id)
        event_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = event_dir / "metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(event_metadata(event), indent=2), encoding="utf-8"
            )

        trades = fetch_trades_for_market(client, market["condition_id"])
        market_path = event_dir / f"{market['market_id']}.json"
        market_path.write_text(json.dumps(trades, indent=2), encoding="utf-8")

        print(
            f"[{idx}/{len(markets)}] saved {len(trades)} trades for market {market['market_id']}"
        )
        if SLEEP_SECONDS:
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
