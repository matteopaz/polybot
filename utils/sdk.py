from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for environments without python-dotenv
    load_dotenv = None

if load_dotenv is not None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)


@dataclass(frozen=True)
class Token:
    """Single outcome token with the most recent price we can find."""

    token_id: str
    outcome: str | None
    price: float | None


@dataclass(frozen=True)
class PricePoint:
    """Point in a token price history."""

    timestamp: dt.datetime
    price: float


@dataclass(frozen=True)
class MakerOrder:
    """Maker leg for a taker trade."""

    order_id: str | None
    maker_address: str | None
    owner: str | None
    matched_amount: float | None
    fee_rate_bps: float | None
    price: float | None
    asset_id: str | None
    outcome: str | None
    side: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Trade:
    """Trade entry returned by the CLOB."""

    id: str | None
    taker_order_id: str | None
    market: str | None
    asset_id: str | None
    side: str | None
    size: float | None
    fee_rate_bps: float | None
    price: float | None
    status: str | None
    match_time: dt.datetime | None
    last_update: dt.datetime | None
    outcome: str | None
    bucket_index: int | None
    owner: str | None
    maker_address: str | None
    maker_orders: list[MakerOrder]
    transaction_hash: str | None
    trader_side: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class PublicTrade:
    """Trade record from the public Data API."""

    proxy_wallet: str | None
    side: str | None
    asset: str | None
    condition_id: str | None
    size: float | None
    price: float | None
    timestamp: dt.datetime | None
    title: str | None
    slug: str | None
    icon: str | None
    event_slug: str | None
    outcome: str | None
    outcome_index: int | None
    name: str | None
    pseudonym: str | None
    bio: str | None
    profile_image: str | None
    profile_image_optimized: str | None
    transaction_hash: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Market:
    """Market metadata from Gamma plus lightweight token wiring."""

    id: str
    question: str | None
    slug: str | None
    description: str | None
    outcomes: list[str]
    outcome_prices: list[float | None]
    clob_token_ids: list[str]
    start_date: dt.datetime | None
    end_date: dt.datetime | None
    created_at: dt.datetime | None
    updated_at: dt.datetime | None
    active: bool | None
    closed: bool | None
    raw: dict[str, Any]

    def tokens(self) -> list[Token]:
        """Pairs outcome labels, token ids, and Gamma prices (if present)."""
        tokens = []
        for idx, token_id in enumerate(self.clob_token_ids):
            outcome = self.outcomes[idx] if idx < len(self.outcomes) else None
            price = self.outcome_prices[idx] if idx < len(self.outcome_prices) else None
            tokens.append(Token(token_id=token_id, outcome=outcome, price=price))
        return tokens


@dataclass(frozen=True)
class Event:
    """Event metadata from Gamma with optional markets parsed in."""

    id: str
    title: str | None
    slug: str | None
    description: str | None
    start_date: dt.datetime | None
    end_date: dt.datetime | None
    created_at: dt.datetime | None
    updated_at: dt.datetime | None
    active: bool | None
    closed: bool | None
    volume: float | None
    liquidity: float | None
    markets: list[Market] | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class L2Credentials:
    """User API credentials for L2-authenticated CLOB calls."""

    api_key: str | None
    api_secret: str | None
    api_passphrase: str | None
    address: str | None


class PolymarketSDK:
    """
    Minimal SDK-style wrapper around Gamma and CLOB endpoints.

    - Gamma: https://gamma-api.polymarket.com (events/markets metadata)
    - CLOB:  https://clob.polymarket.com (prices + orderbook data)
    """

    def __init__(
        self,
        as_of: dt.datetime | dt.date | str | int | float | None = None,
        l2_credentials: L2Credentials | None = None,
    ) -> None:
        self.gamma_base = "https://gamma-api.polymarket.com"
        self.clob_base = "https://clob.polymarket.com"
        self.data_base = "https://data-api.polymarket.com"
        self.as_of = _parse_datetime(as_of)
        self.l2_credentials = l2_credentials
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (polymarket-backtest)",
            }
        )

        self.timeout = 20.0  # seconds

    def set_as_of(
        self, as_of: dt.datetime | dt.date | str | int | float | None
    ) -> PolymarketSDK:
        """Updates the reference date for filtering and price lookups."""
        self.as_of = _parse_datetime(as_of)
        return self

    def set_l2_credentials(self, credentials: L2Credentials | None) -> PolymarketSDK:
        """Stores L2 credentials for authenticated CLOB requests."""
        self.l2_credentials = credentials
        return self

    def list_events(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        id: Sequence[str] | None = None,
        tag_id: int | None = None,
        exclude_tag_id: Sequence[int] | None = None,
        slug: Sequence[str] | None = None,
        tag_slug: str | None = None,
        related_tags: bool | None = None,
        active: bool | None = None,
        archived: bool | None = None,
        featured: bool | None = None,
        cyom: bool | None = None,
        include_chat: bool | None = None,
        include_template: bool | None = None,
        recurrence: str | None = None,
        closed: bool | None = None,
        liquidity_min: float | None = None,
        liquidity_max: float | None = None,
        volume_min: float | None = None,
        volume_max: float | None = None,
        start_date_min: str | None = None,
        start_date_max: str | None = None,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        include_markets: bool = True,
    ) -> list[Event]:
        """Lists events from Gamma with all documented query parameters exposed."""
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": ascending,
            "id": id,
            "tag_id": tag_id,
            "exclude_tag_id": exclude_tag_id,
            "slug": slug,
            "tag_slug": tag_slug,
            "related_tags": related_tags,
            "active": active,
            "archived": archived,
            "featured": featured,
            "cyom": cyom,
            "include_chat": include_chat,
            "include_template": include_template,
            "recurrence": recurrence,
            "closed": closed,
            "liquidity_min": liquidity_min,
            "liquidity_max": liquidity_max,
            "volume_min": volume_min,
            "volume_max": volume_max,
            "start_date_min": start_date_min,
            "start_date_max": start_date_max,
            "end_date_min": end_date_min,
            "end_date_max": end_date_max,
        }
        data = self._get_json(self.gamma_base, "/events", params=params)
        if not isinstance(data, list):
            return []
        return [
            self._event_from_raw(item, include_markets=include_markets)
            for item in _filter_by_date(self.as_of, data, ("createdAt", "creationDate"))
        ]

    def get_event_by_id(self, tid: str, *, include_markets: bool = True) -> Event | None:
        """Fetches a single event by id."""
        data = self._get_json(self.gamma_base, f"/events/{tid}")
        if not isinstance(data, dict):
            return None
        if not _is_visible(self.as_of, data, ("createdAt", "creationDate")):
            return None
        return self._event_from_raw(data, include_markets=include_markets)

    def get_event(self, event_id: str, *, include_markets: bool = True) -> Event | None:
        """Alias for get_event_by_id."""
        return self.get_event_by_id(event_id, include_markets=include_markets)

    def get_event_by_slug(self, slug: str, *, include_markets: bool = True) -> Event | None:
        """Fetches a single event by slug."""
        data = self._get_json(self.gamma_base, f"/events/slug/{slug}")
        if not isinstance(data, dict):
            return None
        if not _is_visible(self.as_of, data, ("createdAt", "creationDate")):
            return None
        return self._event_from_raw(data, include_markets=include_markets)

    def get_event_markets(self, event: Event | str) -> list[Market]:
        """Returns markets for an event, re-fetching if needed."""
        if isinstance(event, Event):
            if event.markets is not None:
                return event.markets
            event_id = event.id
        else:
            event_id = event

        fetched = self.get_event_by_id(event_id, include_markets=True)
        return fetched.markets if fetched and fetched.markets else []

    def list_markets(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        id: Sequence[str] | None = None,
        slug: Sequence[str] | None = None,
        clob_token_ids: Sequence[str] | None = None,
        condition_ids: Sequence[str] | None = None,
        market_maker_address: Sequence[str] | None = None,
        liquidity_num_min: float | None = None,
        liquidity_num_max: float | None = None,
        volume_num_min: float | None = None,
        volume_num_max: float | None = None,
        start_date_min: str | None = None,
        start_date_max: str | None = None,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        tag_id: int | None = None,
        related_tags: bool | None = None,
        cyom: bool | None = None,
        uma_resolution_status: str | None = None,
        game_id: str | None = None,
        sports_market_types: Sequence[str] | None = None,
        rewards_min_size: float | None = None,
        question_ids: Sequence[str] | None = None,
        include_tag: bool | None = None,
        closed: bool | None = None,
    ) -> list[Market]:
        """Lists markets from Gamma with all documented query parameters exposed."""
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": ascending,
            "id": id,
            "slug": slug,
            "clob_token_ids": clob_token_ids,
            "condition_ids": condition_ids,
            "market_maker_address": market_maker_address,
            "liquidity_num_min": liquidity_num_min,
            "liquidity_num_max": liquidity_num_max,
            "volume_num_min": volume_num_min,
            "volume_num_max": volume_num_max,
            "start_date_min": start_date_min,
            "start_date_max": start_date_max,
            "end_date_min": end_date_min,
            "end_date_max": end_date_max,
            "tag_id": tag_id,
            "related_tags": related_tags,
            "cyom": cyom,
            "uma_resolution_status": uma_resolution_status,
            "game_id": game_id,
            "sports_market_types": sports_market_types,
            "rewards_min_size": rewards_min_size,
            "question_ids": question_ids,
            "include_tag": include_tag,
            "closed": closed,
        }
        data = self._get_json(self.gamma_base, "/markets", params=params)
        if not isinstance(data, list):
            return []
        return [
            self._market_from_raw(item)
            for item in _filter_by_date(self.as_of, data, ("createdAt",))
        ]

    def get_market_by_market_id(self, market_id: str) -> Market | None:
        """Fetches a single market by id."""
        data = self._get_json(self.gamma_base, f"/markets/{market_id}")
        if not isinstance(data, dict):
            return None
        if not _is_visible(self.as_of, data, ("createdAt",)):
            return None
        return self._market_from_raw(data)

    def get_market(self, market_id: str) -> Market | None:
        """Alias for get_market_by_market_id."""
        return self.get_market_by_market_id(market_id)

    def get_market_by_slug(self, slug: str) -> Market | None:
        """Fetches a single market by slug."""
        data = self._get_json(self.gamma_base, f"/markets/slug/{slug}")
        if not isinstance(data, dict):
            return None
        if not _is_visible(self.as_of, data, ("createdAt",)):
            return None
        return self._market_from_raw(data)

    def get_token_midpoint(self, token_id: str) -> float | None:
        """
        GET /midpoint (token_id).
        When as_of is set, uses the last trade price from history.
        """
        if self.as_of is not None:
            history = self.get_price_history(token_id)
            return history[-1].price if history else None
        data = self._get_json(self.clob_base, "/midpoint", params={"token_id": token_id})
        mid = data.get("mid") if isinstance(data, dict) else None
        return float(mid) if mid is not None else None

    def get_token_price(self, token_id: str, side: str) -> float | None:
        """
        GET /price (token_id, side). side should be BUY or SELL.
        When as_of is set, uses the last trade price from history.
        """
        if self.as_of is not None:
            history = self.get_price_history(token_id)
            return history[-1].price if history else None
        data = self._get_json(
            self.clob_base, "/price", params={"token_id": token_id, "side": side}
        )
        price = data.get("price") if isinstance(data, dict) else None
        return float(price) if price is not None else None

    def get_price_history(
        self,
        market: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = None,
        fidelity: int | None = None,
    ) -> list[PricePoint]:
        """GET /prices-history for a token id (market), clamped to as_of."""
        if self.as_of is not None:
            as_of_ts = int(self.as_of.timestamp())
            if end_ts is None or end_ts > as_of_ts:
                end_ts = as_of_ts
            if interval is not None:
                seconds = _interval_to_seconds(interval)
                interval = None
                if seconds is not None and start_ts is None:
                    start_ts = max(as_of_ts - seconds, 0)
        params = {
            "market": market,
            "startTs": start_ts,
            "endTs": end_ts,
            "interval": interval,
            "fidelity": fidelity,
        }
        data = self._get_json(self.clob_base, "/prices-history", params=params)
        history = data.get("history") if isinstance(data, dict) else None
        if not history:
            return []
        points = []
        for item in history:
            ts = item.get("t")
            price = item.get("p")
            if ts is None or price is None:
                continue
            if self.as_of is not None and ts > int(self.as_of.timestamp()):
                continue
            points.append(
                PricePoint(
                    timestamp=dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc),
                    price=float(price),
                )
            )
        return sorted(points, key=lambda point: point.timestamp)

    def get_trades(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        taker_only: bool | None = None,
        filter_type: str | None = None,
        filter_amount: float | None = None,
        market: Sequence[str] | None = None,
        event_id: Sequence[int] | None = None,
        user: str | None = None,
        side: str | None = None,
    ) -> list[PublicTrade]:
        """GET /trades from the public Data API (filtered to as_of if set)."""
        params = {
            "limit": limit,
            "offset": offset,
            "takerOnly": taker_only,
            "filterType": filter_type,
            "filterAmount": filter_amount,
            "market": _format_list_param(market),
            "eventId": _format_list_param(event_id),
            "user": user,
            "side": side,
        }
        data = self._get_json(self.data_base, "/trades", params=params)
        if not isinstance(data, list):
            return []
        trades: list[PublicTrade] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            trade = self._public_trade_from_raw(item)
            if self.as_of is not None:
                if trade.timestamp is None or trade.timestamp > self.as_of:
                    continue
            trades.append(trade)
        return trades

    def get_clob_trades(
        self,
        *,
        id: str | None = None,
        maker_address: str | None = None,
        market: str | None = None,
        asset_id: str | None = None,
        before: int | str | None = None,
        after: int | str | None = None,
        next_cursor: str | None = None,
        only_first_page: bool = False,
        max_pages: int | None = None,
        l2_headers: dict[str, str] | None = None,
        l2_credentials: L2Credentials | None = None,
    ) -> list[Trade]:
        """
        GET /data/trades (CLOB, L2 auth required).

        The docs call the maker filter "maker"; official clients send it as
        "maker_address", so that is the parameter exposed here.
        """
        before_value = _parse_int(before)
        after_value = _parse_int(after)
        as_of_ts = int(self.as_of.timestamp()) if self.as_of is not None else None
        if as_of_ts is not None:
            if after_value is not None and after_value > as_of_ts:
                return []
            if before_value is None or before_value > as_of_ts:
                before_value = as_of_ts

        params = {
            "id": id,
            "maker_address": maker_address,
            "market": market,
            "asset_id": asset_id,
            "before": before_value,
            "after": after_value,
        }

        headers = (
            l2_headers
            if l2_headers is not None
            else self._build_l2_headers("/data/trades", l2_credentials=l2_credentials)
        )
        if not headers:
            raise ValueError(
                "Missing L2 headers. Set POLY_API_KEY, POLY_API_SECRET, "
                "POLY_API_PASSPHRASE, POLY_ADDRESS in .env or pass l2_headers."
            )
        results: list[Trade] = []
        cursor = next_cursor or "MA=="
        pages = 0

        while cursor is not None and cursor != "LTE=":
            page_params = dict(params)
            page_params["next_cursor"] = cursor
            data = self._get_json(
                self.clob_base, "/data/trades", params=page_params, headers=headers
            )

            if isinstance(data, dict):
                raw_trades = data.get("data") or []
                cursor = data.get("next_cursor")
            elif isinstance(data, list):
                raw_trades = data
                cursor = None
            else:
                break

            for item in raw_trades:
                if not isinstance(item, dict):
                    continue
                trade = self._trade_from_raw(item)
                if self.as_of is not None:
                    seen = trade.match_time or trade.last_update
                    if seen is not None and seen > self.as_of:
                        continue
                results.append(trade)

            pages += 1
            if only_first_page:
                break
            if max_pages is not None and pages >= max_pages:
                break

        return results

    def get_market_tokens(
        self,
        market: Market | str,
        *,
        price_side: str | None = None,
        price_history_start_ts: int | None = None,
        price_history_end_ts: int | None = None,
        price_history_interval: str | None = None,
        price_history_fidelity: int | None = None,
        fallback_to_outcome_prices: bool = True,
    ) -> list[Token]:
        """
        Returns tokens for a market with prices filled in.

        - If as_of is set (or any price_history_* param is passed), price history
          is used and the last point at or before end_ts is returned.
        - Otherwise, midpoint or side price is used.
        - outcomePrices fallback is only allowed when as_of is None.
        """
        market_obj = market if isinstance(market, Market) else self.get_market(market)
        if market_obj is None:
            return []

        use_history = self.as_of is not None or any(
            [
                price_history_start_ts,
                price_history_end_ts,
                price_history_interval,
                price_history_fidelity,
            ]
        )

        history_end_ts = price_history_end_ts
        if use_history and history_end_ts is None and price_history_interval is None:
            if self.as_of is not None:
                history_end_ts = int(self.as_of.timestamp())

        tokens = []
        allow_fallback = fallback_to_outcome_prices and self.as_of is None
        for idx, token_id in enumerate(market_obj.clob_token_ids):
            outcome = (
                market_obj.outcomes[idx] if idx < len(market_obj.outcomes) else None
            )
            price = None

            if use_history:
                history = self.get_price_history(
                    token_id,
                    start_ts=price_history_start_ts,
                    end_ts=history_end_ts,
                    interval=price_history_interval,
                    fidelity=price_history_fidelity,
                )
                price = history[-1].price if history else None
            else:
                if price_side is not None:
                    price = self.get_token_price(token_id, price_side)
                else:
                    price = self.get_token_midpoint(token_id)

            if price is None and allow_fallback:
                if idx < len(market_obj.outcome_prices):
                    price = market_obj.outcome_prices[idx]

            tokens.append(Token(token_id=token_id, outcome=outcome, price=price))

        return tokens

    def _maker_order_from_raw(self, raw: dict[str, Any]) -> MakerOrder:
        return MakerOrder(
            order_id=_coerce_str(raw.get("order_id")),
            maker_address=raw.get("maker_address"),
            owner=raw.get("owner"),
            matched_amount=_parse_float(raw.get("matched_amount")),
            fee_rate_bps=_parse_float(raw.get("fee_rate_bps")),
            price=_parse_float(raw.get("price")),
            asset_id=raw.get("asset_id"),
            outcome=raw.get("outcome"),
            side=raw.get("side"),
            raw=raw,
        )

    def _trade_from_raw(self, raw: dict[str, Any]) -> Trade:
        match_time = _parse_datetime(raw.get("match_time"))
        last_update = _parse_datetime(raw.get("last_update"))
        status = raw.get("status")
        if self.as_of is not None and last_update is not None and last_update > self.as_of:
            status = None
            last_update = None
        maker_orders_raw = raw.get("maker_orders")
        maker_orders = []
        if isinstance(maker_orders_raw, list):
            maker_orders = [
                self._maker_order_from_raw(item)
                for item in maker_orders_raw
                if isinstance(item, dict)
            ]
        raw_payload = raw
        if self.as_of is not None and last_update is None and raw.get("last_update") is not None:
            raw_payload = dict(raw)
            raw_payload["last_update"] = None
            raw_payload["status"] = status
        return Trade(
            id=_coerce_str(raw.get("id")),
            taker_order_id=raw.get("taker_order_id"),
            market=raw.get("market"),
            asset_id=raw.get("asset_id"),
            side=raw.get("side"),
            size=_parse_float(raw.get("size")),
            fee_rate_bps=_parse_float(raw.get("fee_rate_bps")),
            price=_parse_float(raw.get("price")),
            status=status,
            match_time=match_time,
            last_update=last_update,
            outcome=raw.get("outcome"),
            bucket_index=_parse_int(raw.get("bucket_index")),
            owner=raw.get("owner"),
            maker_address=raw.get("maker_address"),
            maker_orders=maker_orders,
            transaction_hash=raw.get("transaction_hash"),
            trader_side=raw.get("trader_side") or raw.get("type"),
            raw=raw_payload,
        )

    def _public_trade_from_raw(self, raw: dict[str, Any]) -> PublicTrade:
        return PublicTrade(
            proxy_wallet=raw.get("proxyWallet"),
            side=raw.get("side"),
            asset=raw.get("asset"),
            condition_id=raw.get("conditionId"),
            size=_parse_float(raw.get("size")),
            price=_parse_float(raw.get("price")),
            timestamp=_parse_unix_timestamp(raw.get("timestamp")),
            title=raw.get("title"),
            slug=raw.get("slug"),
            icon=raw.get("icon"),
            event_slug=raw.get("eventSlug"),
            outcome=raw.get("outcome"),
            outcome_index=_parse_int(raw.get("outcomeIndex")),
            name=raw.get("name"),
            pseudonym=raw.get("pseudonym"),
            bio=raw.get("bio"),
            profile_image=raw.get("profileImage"),
            profile_image_optimized=raw.get("profileImageOptimized"),
            transaction_hash=raw.get("transactionHash"),
            raw=raw,
        )

    def _event_from_raw(self, raw: dict[str, Any], *, include_markets: bool) -> Event:
        markets = None
        if include_markets and isinstance(raw.get("markets"), list):
            markets = [
                self._market_from_raw(item)
                for item in _filter_by_date(self.as_of, raw["markets"], ("createdAt",))
            ]
        active = raw.get("active")
        closed = raw.get("closed")
        volume = _parse_float(raw.get("volume"))
        liquidity = _parse_float(raw.get("liquidity"))
        updated_at = _parse_datetime(raw.get("updatedAt"))
        if self.as_of is not None:
            active = None
            closed = None
            volume = None
            liquidity = None
            updated_at = None
            raw = {
                "id": str(raw.get("id")),
                "title": raw.get("title"),
                "slug": raw.get("slug"),
                "description": raw.get("description"),
                "startDate": raw.get("startDate"),
                "endDate": raw.get("endDate"),
                "createdAt": raw.get("createdAt"),
                "creationDate": raw.get("creationDate"),
                "markets": None,
            }
        return Event(
            id=str(raw.get("id")),
            title=raw.get("title"),
            slug=raw.get("slug"),
            description=raw.get("description"),
            start_date=_parse_datetime(raw.get("startDate")),
            end_date=_parse_datetime(raw.get("endDate")),
            created_at=_parse_datetime(raw.get("createdAt") or raw.get("creationDate")),
            updated_at=updated_at,
            active=active,
            closed=closed,
            volume=volume,
            liquidity=liquidity,
            markets=markets,
            raw=raw,
        )

    def _market_from_raw(self, raw: dict[str, Any]) -> Market:
        outcomes = _parse_json_list(raw.get("outcomes"))
        clob_token_ids = _parse_json_list(raw.get("clobTokenIds"))
        outcome_prices = [
            _parse_float(item) for item in _parse_json_list(raw.get("outcomePrices"))
        ]
        if self.as_of is not None:
            outcome_prices = self._prices_as_of(clob_token_ids)
            if len(outcome_prices) < len(outcomes):
                outcome_prices.extend([None for _ in range(len(outcomes) - len(outcome_prices))])
        active = raw.get("active")
        closed = raw.get("closed")
        updated_at = _parse_datetime(raw.get("updatedAt"))
        if self.as_of is not None:
            active = None
            closed = None
            updated_at = None
            raw = {
                "id": str(raw.get("id")),
                "question": raw.get("question"),
                "slug": raw.get("slug"),
                "description": raw.get("description"),
                "outcomes": raw.get("outcomes"),
                "clobTokenIds": raw.get("clobTokenIds"),
                "startDate": raw.get("startDate"),
                "endDate": raw.get("endDate"),
                "createdAt": raw.get("createdAt"),
            }
        return Market(
            id=str(raw.get("id")),
            question=raw.get("question"),
            slug=raw.get("slug"),
            description=raw.get("description"),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            start_date=_parse_datetime(raw.get("startDate")),
            end_date=_parse_datetime(raw.get("endDate")),
            created_at=_parse_datetime(raw.get("createdAt")),
            updated_at=updated_at,
            active=active,
            closed=closed,
            raw=raw,
        )

    def _prices_as_of(self, token_ids: Sequence[str]) -> list[float | None]:
        """Fetches last known prices for each token at or before as_of."""
        if self.as_of is None:
            return []
        prices: list[float | None] = []
        for token_id in token_ids:
            history = self.get_price_history(token_id)
            prices.append(history[-1].price if history else None)
        return prices

    def _get_json(
        self,
        base: str,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        if params:
            params = {key: value for key, value in params.items() if value is not None}
        try:
            resp = self.session.get(
                url, params=params, headers=headers, timeout=self.timeout
            )
            if resp.status_code in (400, 404):
                return {}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            raise

    def _build_l2_headers(
        self,
        request_path: str,
        *,
        method: str = "GET",
        body: Any | None = None,
        l2_credentials: L2Credentials | None = None,
    ) -> dict[str, str]:
        creds = l2_credentials or self.l2_credentials or _load_l2_credentials_from_env()
        if creds is None:
            return {}
        if not all(
            [
                creds.api_key,
                creds.api_secret,
                creds.api_passphrase,
                creds.address,
            ]
        ):
            raise ValueError(
                "L2 credentials require api_key, api_secret, api_passphrase, address. "
                "Set POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE, POLY_ADDRESS in .env."
            )
        timestamp = int(time.time())
        signature = _build_hmac_signature(
            creds.api_secret,
            timestamp,
            method.upper(),
            request_path,
            body,
        )
        return {
            "POLY_ADDRESS": creds.address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": str(timestamp),
            "POLY_API_KEY": creds.api_key,
            "POLY_PASSPHRASE": creds.api_passphrase,
        }


def _parse_json_list(value: Any) -> list[str]:
    """Gamma returns some arrays as JSON strings; normalize them to list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        cleaned = text.strip("[]")
        return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()]
    return [str(value)]


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _load_l2_credentials_from_env() -> L2Credentials | None:
    api_key = os.getenv("POLY_API_KEY")
    api_secret = os.getenv("POLY_API_SECRET")
    api_passphrase = os.getenv("POLY_API_PASSPHRASE")
    address = os.getenv("POLY_ADDRESS")
    if not any([api_key, api_secret, api_passphrase, address]):
        return None
    return L2Credentials(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        address=address,
    )


def _parse_datetime(value: Any | None) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime.combine(value, dt.time.min)
    elif isinstance(value, (int, float)):
        parsed = dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _parse_unix_timestamp(value: Any | None) -> dt.datetime | None:
    if value is None:
        return None
    ts = _parse_float(value)
    if ts is None:
        return None
    if ts > 1_000_000_000_000:
        ts = ts / 1000
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)


def _is_visible(as_of: dt.datetime | None, item: dict[str, Any], date_keys: tuple[str, ...]) -> bool:
    if as_of is None:
        return True
    for key in date_keys:
        seen = _parse_datetime(item.get(key))
        if seen is not None:
            return seen <= as_of
    return False


def _filter_by_date(
    as_of: dt.datetime | None, items: Iterable[dict[str, Any]], date_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    if as_of is None:
        return list(items)
    return [item for item in items if _is_visible(as_of, item, date_keys)]


def _format_list_param(values: Sequence[Any] | None) -> str | None:
    if not values:
        return None
    return ",".join(str(value) for value in values)

def _interval_to_seconds(interval: str) -> int | None:
    if interval == "1h":
        return 60 * 60
    if interval == "6h":
        return 6 * 60 * 60
    if interval == "1d":
        return 24 * 60 * 60
    if interval == "1w":
        return 7 * 24 * 60 * 60
    if interval == "1m":
        return 30 * 24 * 60 * 60
    return None


def _build_hmac_signature(
    secret: str, timestamp: int, method: str, request_path: str, body: Any | None = None
) -> str:
    base64_secret = base64.urlsafe_b64decode(secret)
    message = f"{timestamp}{method}{request_path}"
    if body is not None:
        message += str(body).replace("'", '"')
    digest = hmac.new(base64_secret, message.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")
