"""
LLM Module for C.O.O.K.
CPSC 481 - Artificial Intelligence

Two LLM-powered features:
1. Estimate grocery costs for recipes (professor's suggestion)
2. Generate natural language explanations for recommendations

Compatible with OpenAI API format (works with OpenAI, NRP Managed LLMs, etc.)
"""

import os
import json
from openai import OpenAI

# ──────────────────────────────────────────────
# CONFIG — set your API key and base URL here
# ──────────────────────────────────────────────
# Option A: OpenAI
#   OPENAI_API_KEY=sk-... (set as environment variable)
#
# Option B: NRP Managed LLMs (free for CSUF students)
#   Set LLM_BASE_URL and LLM_API_KEY environment variables
#
# Option C: No API key — falls back to static estimates

def get_client():
    """Return an OpenAI-compatible client, or None if no key is set."""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")  # For NRP or other providers

    if not api_key:
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")


# ──────────────────────────────────────────────
# 1. COST ESTIMATION
# ──────────────────────────────────────────────
def estimate_recipe_cost(recipe_name: str, ingredients: list[str]) -> dict | None:
    """
    Ask the LLM to estimate the grocery cost of a recipe
    based on average US grocery prices.

    Returns: {"total_estimate": float, "per_ingredient": {name: price}}
    Or None if LLM is unavailable.
    """
    client = get_client()
    if not client:
        return None

    ingredient_list = ", ".join(ingredients)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grocery price estimator. Given a recipe name and "
                        "ingredient list, estimate the cost of each ingredient based on "
                        "average US grocery store prices (assuming standard portions for "
                        "one meal serving 2-4 people). Respond ONLY with valid JSON in "
                        "this exact format, no other text:\n"
                        '{"total_estimate": 12.50, "per_ingredient": {"chicken": 4.50, "rice": 1.20}}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Recipe: {recipe_name}\nIngredients: {ingredient_list}",
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )

        text = response.choices[0].message.content.strip()
        # Clean markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except Exception as e:
        print(f"[LLM] Cost estimation failed: {e}")
        return None


# ──────────────────────────────────────────────
# 2. RECOMMENDATION EXPLANATION
# ──────────────────────────────────────────────
def explain_recommendation(
    recipe_name: str,
    bn_score: float,
    pantry_match: str,
    ingredients_owned: int,
    ingredients_needed: list[str],
    price_estimate: float,
    user_budget: float,
    cuisine: str,
    cook_time: str,
) -> str:
    """
    Generate a natural language explanation of why a recipe
    was recommended, based on the Bayesian network output.

    Returns a 1-2 sentence explanation, or a static fallback.
    """
    client = get_client()

    if not client:
        # Static fallback — no LLM needed
        return _static_explanation(
            recipe_name, bn_score, pantry_match,
            ingredients_owned, ingredients_needed,
            price_estimate, user_budget, cuisine, cook_time
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful cooking assistant. Given data about a recipe "
                        "recommendation, write a brief 1-2 sentence explanation of why "
                        "this recipe is a good match for the user. Be friendly and concise. "
                        "Mention the key factors: pantry match, budget fit, and cook time."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recipe: {recipe_name} ({cuisine} cuisine)\n"
                        f"Match score: {bn_score * 100:.0f}%\n"
                        f"Pantry match: {pantry_match} ({ingredients_owned} ingredients you have)\n"
                        f"Still need to buy: {', '.join(ingredients_needed) if ingredients_needed else 'nothing'}\n"
                        f"Estimated cost: ${price_estimate:.2f} (your budget: ${user_budget:.2f})\n"
                        f"Cook time: {cook_time}"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM] Explanation failed: {e}")
        return _static_explanation(
            recipe_name, bn_score, pantry_match,
            ingredients_owned, ingredients_needed,
            price_estimate, user_budget, cuisine, cook_time
        )


# ──────────────────────────────────────────────
# 3. RECIPE SUGGESTION (professor's approach #2)
# ──────────────────────────────────────────────
def llm_recommend(
    recipes: list[dict],
    user_pantry: list[str],
    user_budget: float,
    cuisine_pref: str | None = None,
    diet_pref: str = "none",
) -> str:
    """
    Give the LLM the full recipe dataset + user constraints
    and ask it to recommend the best option.

    This implements professor's suggested approach #2.
    Returns the LLM's recommendation as text, or None if unavailable.
    """
    client = get_client()
    if not client:
        return None

    # Build a compact recipe summary for the prompt
    recipe_summary = []
    for r in recipes:
        recipe_summary.append(
            f"- {r['name']} | {r['cuisine']} | {r['diet']} | "
            f"${r['price_estimate']:.2f} | {r['cook_time']} | "
            f"Ingredients: {', '.join(r['ingredients'])}"
        )
    recipe_text = "\n".join(recipe_summary)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meal planning assistant. Given a list of recipes and "
                        "the user's constraints, recommend the top 3 best recipes and "
                        "explain why each is a good fit. Consider: how many ingredients "
                        "the user already has, whether the recipe fits their budget, "
                        "cuisine preference, and diet. Be concise and practical."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"My pantry: {', '.join(user_pantry)}\n"
                        f"My budget: ${user_budget:.2f}\n"
                        f"Cuisine preference: {cuisine_pref or 'any'}\n"
                        f"Diet: {diet_pref}\n\n"
                        f"Available recipes:\n{recipe_text}"
                    ),
                },
            ],
            temperature=0.5,
            max_tokens=400,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM] Recommendation failed: {e}")
        return None


# ──────────────────────────────────────────────
# STATIC FALLBACK (no API key needed)
# ──────────────────────────────────────────────
def _static_explanation(
    recipe_name, bn_score, pantry_match,
    ingredients_owned, ingredients_needed,
    price_estimate, user_budget, cuisine, cook_time
):
    """Generate explanation without LLM — template-based."""

    parts = []

    # Pantry match
    if pantry_match == "high":
        parts.append(f"You already have {ingredients_owned} of the ingredients")
    elif pantry_match == "medium":
        parts.append(f"You have some of the ingredients on hand")
    else:
        parts.append(f"You'll need most ingredients for this one")

    # Budget
    if price_estimate <= user_budget * 0.6:
        parts.append(f"and it's well under your ${user_budget:.0f} budget at ${price_estimate:.2f}")
    elif price_estimate <= user_budget:
        parts.append(f"and it fits your ${user_budget:.0f} budget at ${price_estimate:.2f}")

    # Cook time
    if cook_time == "short":
        parts.append("plus it's quick to make (under 30 min)")

    # What to buy
    if ingredients_needed and len(ingredients_needed) <= 3:
        parts.append(f"— just pick up {', '.join(ingredients_needed)}")

    return " ".join(parts[:2]) + "." if parts else f"A solid {cuisine} option within your budget."


# ──────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  C.O.O.K. — LLM Module Demo")
    print("=" * 60)

    # Check if LLM is available
    client = get_client()
    if client:
        print(f"\n  ✓ LLM connected (model: {MODEL})")
    else:
        print("\n  ✗ No API key found — using static fallback")
        print("    Set OPENAI_API_KEY or LLM_API_KEY to enable LLM features")

    # Demo: static explanation (always works)
    print("\n" + "─" * 60)
    print("  DEMO: Recommendation Explanation")
    print("─" * 60)

    explanation = explain_recommendation(
        recipe_name="Chicken Stir Fry",
        bn_score=0.83,
        pantry_match="high",
        ingredients_owned=5,
        ingredients_needed=["broccoli"],
        price_estimate=7.50,
        user_budget=15.00,
        cuisine="asian",
        cook_time="short",
    )
    print(f"\n  Recipe: Chicken Stir Fry")
    print(f"  Explanation: {explanation}")

    explanation2 = explain_recommendation(
        recipe_name="Black Bean Burritos",
        bn_score=0.88,
        pantry_match="high",
        ingredients_owned=3,
        ingredients_needed=["sour cream", "salsa", "cheese"],
        price_estimate=5.50,
        user_budget=10.00,
        cuisine="mexican",
        cook_time="short",
    )
    print(f"\n  Recipe: Black Bean Burritos")
    print(f"  Explanation: {explanation2}")

    # Demo: LLM cost estimation (only if API key is set)
    if client:
        print("\n" + "─" * 60)
        print("  DEMO: LLM Cost Estimation")
        print("─" * 60)
        cost = estimate_recipe_cost(
            "Chicken Stir Fry",
            ["chicken", "rice", "soy sauce", "broccoli", "garlic", "oil"]
        )
        if cost:
            print(f"\n  Total estimate: ${cost['total_estimate']:.2f}")
            for item, price in cost.get("per_ingredient", {}).items():
                print(f"    {item}: ${price:.2f}")

        print("\n" + "─" * 60)
        print("  DEMO: LLM Direct Recommendation")
        print("─" * 60)

        from recipe_recommender import RECIPES
        result = llm_recommend(
            RECIPES,
            user_pantry=["chicken", "rice", "garlic", "soy sauce"],
            user_budget=15.00,
            cuisine_pref="asian",
        )
        if result:
            print(f"\n{result}")


if __name__ == "__main__":
    main()
