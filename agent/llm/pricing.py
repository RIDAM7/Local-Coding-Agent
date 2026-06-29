"""Cost / token price table (Phase 7C).

Prices are USD per 1,000 tokens (input, output). Matching is by substring of the
model name within the provider, longest/most-specific keys listed first. Local
providers (``ollama``) are always free. Unknown cloud models price at 0.0 (we do
not guess), so telemetry never invents a number.
"""

# provider -> list of (model_keyword, usd_per_1k_input, usd_per_1k_output)
PRICES = {
    "openai": [
        ("gpt-4o-mini", 0.00015, 0.0006),
        ("gpt-4o", 0.0025, 0.01),
        ("gpt-4-turbo", 0.01, 0.03),
        ("gpt-4", 0.03, 0.06),
        ("gpt-3.5", 0.0005, 0.0015),
        ("o1-mini", 0.003, 0.012),
        ("o1", 0.015, 0.06),
    ],
    "anthropic": [
        ("haiku", 0.00025, 0.00125),
        ("sonnet", 0.003, 0.015),
        ("opus", 0.015, 0.075),
    ],
    "google": [
        ("flash", 0.000075, 0.0003),
        ("pro", 0.00125, 0.005),
    ],
}


def price_for(provider: str, model: str):
    """Return (usd_per_1k_input, usd_per_1k_output) for a provider+model, or (0,0)."""
    provider = (provider or "").lower()
    model = (model or "").lower()
    for keyword, price_in, price_out in PRICES.get(provider, []):
        if keyword in model:
            return price_in, price_out
    return 0.0, 0.0


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost for a call. Local providers always return 0.0."""
    if (provider or "").lower() == "ollama":
        return 0.0
    price_in, price_out = price_for(provider, model)
    cost = (input_tokens / 1000.0) * price_in + (output_tokens / 1000.0) * price_out
    return round(cost, 6)
