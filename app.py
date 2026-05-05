from flask import Flask, render_template, request
from recipe_recommender import build_network, recommend_recipes, RECIPES

app = Flask(__name__)

VALID_CUISINES = {"any", "asian", "italian", "mexican", "american"}
VALID_DIETS = {"none", "vegetarian"}

model = build_network()

RECIPE_INGREDIENTS = {r["name"]: r["ingredients"] for r in RECIPES}

RECIPE_DESCRIPTIONS = {
    "Chicken Stir Fry": "Tender chicken with crisp broccoli and garlic tossed in a savory soy sauce.",
    "Spaghetti Bolognese": "Classic Italian meat sauce slow-cooked with tomatoes and herbs over pasta.",
    "Veggie Buddha Bowl": "Wholesome quinoa bowl loaded with roasted veggies and tahini dressing.",
    "Beef Tacos": "Seasoned ground beef in warm tortillas with fresh lettuce, cheese, and salsa.",
    "Salmon Teriyaki": "Glazed salmon fillet with teriyaki sauce served over steamed rice and broccoli.",
    "Margherita Pizza (Homemade)": "Crispy homemade crust topped with fresh mozzarella, tomato sauce, and basil.",
    "Shrimp Alfredo": "Creamy parmesan pasta tossed with garlic butter shrimp.",
    "Black Bean Burritos": "Hearty black bean and rice burritos loaded with cheese, salsa, and sour cream.",
    "Grilled Chicken Salad": "Light grilled chicken over crisp greens with cucumber, tomato, and lemon.",
    "Pad Thai": "Classic Thai stir-fried rice noodles with shrimp, egg, peanuts, and lime.",
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        error=None,
        budget="",
        pantry="",
        cuisine="any",
        diet="none",
    )


@app.route("/results", methods=["POST"])
def results():
    budget_raw = request.form.get("budget", "").strip()
    pantry_raw = request.form.get("pantry", "").strip()
    cuisine = request.form.get("cuisine", "any").strip().lower()
    diet = request.form.get("diet", "none").strip().lower()

    error = None

    if not budget_raw:
        error = "Please enter a budget."
    elif not pantry_raw:
        error = "Please enter at least one pantry ingredient."
    elif cuisine not in VALID_CUISINES:
        error = "Please choose a valid cuisine."
    elif diet not in VALID_DIETS:
        error = "Please choose a valid diet preference."

    budget = None
    if error is None:
        try:
            budget = float(budget_raw)
            if budget <= 0:
                error = "Budget must be greater than 0."
        except ValueError:
            error = "Budget must be a valid number."

    if error:
        return render_template(
            "index.html",
            error=error,
            budget=budget_raw,
            pantry=pantry_raw,
            cuisine=cuisine,
            diet=diet,
        )

    pantry = [item.strip().lower() for item in pantry_raw.split(",") if item.strip()]

    cuisine_pref = None if cuisine == "any" else cuisine

    top_recipes = recommend_recipes(
        model,
        user_pantry=pantry,
        user_budget=budget,
        cuisine_pref=cuisine_pref,
        diet_pref=diet,
    )

    top_recipes = [r for r in top_recipes if r["ingredients_owned"] > 0]
    if cuisine_pref:
        top_recipes = [r for r in top_recipes if r["cuisine"] == cuisine_pref]

    for r in top_recipes:
        r["ingredients"] = RECIPE_INGREDIENTS.get(r["recipe"], [])
        r["description"] = RECIPE_DESCRIPTIONS.get(r["recipe"], "")

    return render_template(
        "results.html",
        recipes=top_recipes,
        budget=f"{budget:.2f}",
        pantry=", ".join(pantry),
        cuisine=cuisine,
        diet=diet,
    )


if __name__ == "__main__":
    app.run(debug=True)
