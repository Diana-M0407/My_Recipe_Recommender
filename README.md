# C.O.O.K.
## A Budget-Aware Recipe Recommendation System Using Bayesian Networks and Generative AI

C.O.O.K. (Cost Optimized Online Kitchen) is an AI-driven recipe recommendation web application designed to help users discover meals based on:
- Budget constraints
- Pantry ingredients
- Cuisine preferences
- Dietary preferences
- Cook time preferences

The system combines Bayesian Networks, probabilistic inference, and Generative AI to produce personalized recipe recommendations and natural-language explanations.

---

# Table of Contents

1. Project Overview
2. Features
3. Technologies Used
4. System Architecture
5. Bayesian Network
6. Project Structure
7. File and Folder Descriptions
8. Algorithms and AI Components
9. Data
10. How to Run the Project
11. Example Workflow
12. Future Improvements
13. References
14. Authors

---

# Project Overview

Meal planning can be difficult for individuals attempting to balance food costs, dietary preferences, available ingredients, and cooking time. Existing recipe recommendation systems often focus only on user preferences and do not simultaneously consider grocery cost, pantry utilization, dietary restrictions, and cooking time.

This project addresses the problem by developing an AI-driven recommendation system that combines Bayesian Networks and probabilistic reasoning to generate personalized, budget-aware recipe recommendations.

The system evaluates recipes based on:
- pantry ingredient overlap
- budget compatibility
- cuisine preference
- dietary restrictions
- cook time preference

Recipes are then ranked according to recommendation probability and estimated additional grocery cost.

---

# Features

- Budget-aware recipe recommendations
- Pantry ingredient matching
- Bayesian probabilistic inference
- Recipe ranking system
- Natural-language recommendation explanations using LLMs
- Flask-based web interface
- User-friendly input form and results page
- Budget classification
- Cook-time filtering
- Personalized recommendation probabilities

---

# Technologies Used

## Programming Languages
- Python
- HTML/CSS
- Jinja2 Templates

## Libraries
- Flask
- pgmpy
- pandas
- numpy
- OpenAI API client

## AI Concepts
- Bayesian Networks
- Probabilistic Inference
- Variable Elimination
- Generative AI
- Large Language Models (LLMs)

---

# System Architecture

The application consists of five major components:

1. User / Browser
2. Frontend (Flask + Jinja2)
3. Backend Flask Application
4. Recommendation Engine (AI Layer)
5. Recipe Database

General workflow:
1. User enters pantry ingredients and preferences
2. Backend validates the input
3. Recommendation engine computes recommendation probabilities
4. Recipes are ranked
5. Results are displayed to the user
6. LLM module generates recommendation explanations

---

# Bayesian Network

The recommendation engine uses a Bayesian Network implemented with the pgmpy library.

## Bayesian Network Variables

Parent/Input Nodes:
- CuisinePref
- DietPref
- Budget
- CookTime
- PantryMatch

Child/Output Node:
- Recommend

The system computes:

P(Recommend = yes | evidence)

for each recipe in the database.

## Inference Method

The system uses:
- Variable Elimination
- Conditional Probability Tables (CPTs)

to compute recommendation probabilities dynamically based on user evidence.

---

# Project Structure

```text
COOK/
│
├── app.py
├── recipe_recommender.py
├── llm_helper.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   ├── style.css
│   └── images/
│
└── data/
    └── recipes.json
```

---

# File and Folder Descriptions

## app.py

Main Flask application.

Responsibilities:
- Handles HTTP routes
- Processes user input
- Validates form data
- Calls the recommendation engine
- Sends recommendation results to the frontend

This file acts as the controller between the frontend and backend AI logic.

---

## recipe_recommender.py

Core recommendation engine.

Responsibilities:
- Builds the Bayesian Network
- Defines Conditional Probability Tables (CPTs)
- Computes pantry overlap
- Classifies budget levels
- Performs probabilistic inference using Variable Elimination
- Ranks recipes by recommendation probability

This is the primary AI component of the project.

---

## llm_helper.py

Handles Generative AI functionality.

Responsibilities:
- Connects to the OpenAI-compatible API
- Generates natural-language recommendation explanations
- Produces optional grocery cost summaries
- Provides fallback responses if no API key is available

---

## requirements.txt

Contains all required Python dependencies needed to run the project.

Example dependencies:
- Flask
- pgmpy
- pandas
- numpy
- openai

---

# templates/

Contains HTML frontend pages rendered using Jinja2 templates.

## index.html

Homepage input form where users enter:
- pantry ingredients
- budget
- cuisine preference
- diet preference
- cook time preference

## results.html

Displays:
- ranked recipe recommendations
- recommendation probability
- pantry match information
- generated explanations

---

# static/

Contains static frontend assets.

## style.css

Custom CSS styling for the application UI.

## images/

Stores:
- screenshots
- diagrams
- frontend assets

---

# data/

Contains project data files.

## recipes.json

Stores the manually created recipe database used by the recommendation engine.

Recipe entries include:
- ingredients
- cuisine labels
- dietary categories
- cook time classifications
- estimated pricing

NOTE:
Only manually collected project data is included in this repository.

Public libraries, APIs, and external datasets are NOT redistributed.

External resources are referenced in the final report and documentation.

---

# Algorithms and AI Components

## Pantry Matching

Pantry matching measures the overlap between user ingredients and recipe ingredients.

The overlap ratio determines whether the pantry match is classified as:
- low
- medium
- high

Higher pantry overlap increases recommendation probability and reduces additional grocery cost.

---

## Budget Classification

Recipes are classified into:
- low budget
- medium budget
- high budget

based on estimated ingredient costs.

---

## Bayesian Probabilistic Inference

The primary algorithm used in the system is Bayesian probabilistic inference through Variable Elimination provided by the pgmpy library.

Recipes are evaluated using:
- pantry overlap
- budget level
- cuisine preference
- diet preference
- cook time

Recipes are then ranked according to:
- recommendation probability
- estimated grocery cost

---

## LLM Module

An additional LLM module was integrated using the OpenAI-compatible API format.

The module provides:
1. Natural-language explanations for recipe recommendations
2. Optional grocery cost estimation
3. Experimental direct LLM-based recipe recommendation

The system also includes a static fallback explanation generator when no API key is available.

---

# Data

The system uses a manually constructed recipe database containing:
- 18 recipes
- cuisine labels
- dietary categories
- ingredient lists
- cook time categories
- estimated pricing

Recipe data is stored locally using:
- Python dictionaries
- JSON-style structures

## Data Policy

Only manually collected project data is included.

Public datasets and external libraries are NOT redistributed in this repository.

External resources are referenced through official links in the report and README.

---

# How to Run the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run the Flask Application

```bash
python app.py
```

## 3. Open in Browser

```text
http://127.0.0.1:5000
```

---

# Example Workflow

Example user input:
- Pantry: rice, garlic, chicken
- Budget: medium
- Cuisine: Asian
- Diet: none
- Cook Time: short

Example system process:
1. Flask backend validates input
2. Pantry overlap is computed
3. Bayesian evidence is generated
4. Variable Elimination computes recommendation probabilities
5. Recipes are ranked
6. Results are displayed
7. LLM generates recommendation explanations

Example output:
- Chicken Fried Rice
- Pantry Match: High
- Recommendation Probability: 0.87

---

# Future Improvements

Potential future improvements include:
- Real grocery pricing APIs
- Persistent user accounts
- Learning from user feedback
- AWS cloud deployment
- Larger recipe datasets
- Collaborative filtering
- Improved CPT learning from real user data
- Multimodal ingredient recognition using images
- Voice assistant integration

---

# References

- Flask Documentation  
  https://flask.palletsprojects.com/

- pgmpy Documentation  
  https://pgmpy.org

- OpenAI API Documentation  
  https://platform.openai.com/docs

- Russell, S., & Norvig, P.  
  Artificial Intelligence: A Modern Approach (4th ed.)

---

# Authors

- Diana Maldonado
- Reign Pierson
- Leiss Amini

California State University, Fullerton  
CPSC 481 - Artificial Intelligence
