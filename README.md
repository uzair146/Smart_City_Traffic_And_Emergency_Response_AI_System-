# Smart City Traffic and Emergency Response AI System

A Python-based AI system that manages city traffic, handles emergency vehicle routing, and enforces traffic policies using multiple AI techniques.

---

## What This Project Does

This system takes incoming requests from vehicles (civilian or emergency) and decides the best route, checks 
if the request is allowed under city policy, assigns traffic signal phases, and predicts priority levels — all automatically.

---

## Modules

| Module | Name | What It Does |
|--------|------|--------------|
| 1 | Visualization | Draws the city map and highlights the recommended route in red |
| 2 | Data Preprocessing | Cleans and validates incoming requests, converts text to numbers |
| 3 | ANN Priority Prediction | Uses a neural network to predict request priority (low/normal/high/critical) |
| 4 | CSP Signal Control | Assigns traffic signal phases to intersections without conflicts |
| 5 | Search and Navigation | Finds the best route using BFS, UCS, or A* depending on request type |
| 6 | FOL Validation | Checks city policy using First Order Logic rules (approved/rejected) |
| 7 | Request Router | Directs each request to the right modules based on its category |
| 8 | Final Response | Collects all outputs and prints a clean, readable response |

---

## Request Types

- **Route_Request** — Find a path from A to B (uses BFS)
- **Policy_Check** — Check if a vehicle action is allowed (uses FOL)
- **Control_Allocation_Request** — Assign traffic signals (uses CSP + FOL)
- **Emergency_Response_Request** — Full pipeline: priority prediction, validation, signals, and routing (uses ANN + FOL + CSP + A*)
- **Integrated_City_Service_Request** — Same as emergency but for integrated city services

---

## AI Techniques Used

- **BFS** (Breadth First Search) — shortest path in unweighted graph
- **UCS** (Uniform Cost Search) — cheapest path by travel time
- **A\*** — fastest path using heuristic estimates (used for emergencies)
- **ANN** (Artificial Neural Network) — predicts priority level using MLPClassifier
- **CSP** (Constraint Satisfaction Problem) — assigns signal phases with MRV + forward checking
- **FOL** (First Order Logic) — validates requests against city traffic policy rules

---

## How to Run

**1. Install dependencies**
```bash
pip install matplotlib networkx scikit-learn numpy
```

> You also need the `aima3` library for FOL support (`logic.py` and `utils.py`)
```bash
pip install aima3
```

**2. Run the notebook**

Open `Project_Code.ipynb` in Jupyter Notebook or VS Code and run all cells.

---

## Sample Output

The system processes 4 test requests and for each one prints:

```
FINAL RESPONSE: REQ-004

Request ID:            REQ-004
Category:              Emergency_Response_Request
Vehicle:               Emergency
Predicted Priority:    CRITICAL
Policy Status:         Approved
Assigned Control Plan: North_Station: PhaseA, Central_Junction: PhaseB, ...
Recommended Route:     North_Station -> Central_Junction -> City_Hospital
Travel Estimate:       10 Mins (A*)

Decision Message: Request APPROVED. Operating under CRITICAL priority.
                  Emergency corridor and signal overrides have been secured.
                  Navigation path is ready.
```

Then a visual map is shown with the route highlighted in red.

---

## Group Members


| Muhammad Uzair Hussain   |
| Muhammad Abdullah        |
| Masroor Ahmad Zafar      |

---

## Course

Artificial Intelligence — Spring 2026
