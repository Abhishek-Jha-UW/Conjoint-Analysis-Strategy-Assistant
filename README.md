# Conjoint Analysis Tool

An app that helps you run an **AI-assisted conjoint analysis** to understand **which product attributes matter most** (and how trade-offs change with price).

## What you can do

- Define a product and compare **attribute importance** and **price sensitivity**
- Run a conjoint-style analysis and explore the results
- Get an AI-written interpretation and potential strategy ideas (based on computed results)

## How attributes and price points are created

You have two ways to set up the study:

1. **Manual mode**
   - You enter the attributes and levels yourself (including price points).

2. **AI-assisted mode**
   - You type a product name/description.
   - AI proposes reasonable attributes and price points based on its prior knowledge (and optional web-sourced context if you provide it).

## Notes

- AI-generated inputs are **suggestions**. Always review attributes/levels for realism and bias before trusting outcomes.
- The app is designed for fast iteration and portfolio demos; synthetic or AI-assisted data is not a substitute for real customer research.

## Planned files

- `app.py`: Streamlit UI
- `model.py`: conjoint design, estimation, and simulations
- `requirements.txt`: dependencies

