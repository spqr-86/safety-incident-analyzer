"""Tests for intent_gate node."""

from __future__ import annotations

import pytest

from src.v7.nodes.intent_gate import intent_gate, route_by_intent


class TestIntentGate:
    @pytest.mark.unit
    def test_noise_short_query(self):
        result = intent_gate({"query": "hi"})
        assert result["intent"] == "noise"

    @pytest.mark.unit
    def test_noise_greeting(self):
        result = intent_gate({"query": "привет"})
        assert result["intent"] == "noise"

    @pytest.mark.unit
    def test_domain_query(self):
        result = intent_gate({"query": "Требования к ограждениям лестниц"})
        assert result["intent"] == "domain"

    @pytest.mark.unit
    def test_empty_query(self):
        result = intent_gate({"query": ""})
        assert result["intent"] == "noise"

    @pytest.mark.unit
    def test_none_query(self):
        result = intent_gate({})
        assert result["intent"] == "noise"


class TestRouteByIntent:
    @pytest.mark.unit
    def test_noise_routes_to_end(self):
        assert route_by_intent({"intent": "noise"}) == "end"

    @pytest.mark.unit
    def test_domain_routes_to_router(self):
        assert route_by_intent({"intent": "domain"}) == "router"
