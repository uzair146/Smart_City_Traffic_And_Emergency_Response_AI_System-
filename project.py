# Importing required libraries for search algorithms and data structures
import heapq
from collections import deque

# Importing utility functions and logic framework for FOL validation
from utils import *
from logic import *

# Importing numpy and neural network for priority prediction
import numpy as np
from sklearn.neural_network import MLPClassifier

# Importing matplotlib and networkx for route visualization
import matplotlib.pyplot as plt
import networkx as nx

# Graph without weights used for simple BFS path finding
unweighted_city_graph = {
    'Police_HQ': ['Traffic_Control_Center', 'River_Bridge'],
    'Traffic_Control_Center': ['Police_HQ', 'North_Station'],
    'North_Station': ['Traffic_Control_Center', 'River_Bridge', 'Central_Junction'],
    'River_Bridge': ['Police_HQ', 'North_Station', 'Stadium'],
    'Stadium': ['River_Bridge', 'Airport_Road', 'East_Market'],
    'Airport_Road': ['Stadium', 'South_Residential'],
    'South_Residential': ['Airport_Road', 'Central_Junction', 'City_Hospital'],
    'East_Market': ['Stadium', 'Central_Junction', 'City_Hospital'],
    'Central_Junction': ['North_Station', 'South_Residential', 'East_Market', 'West_Terminal'],
    'West_Terminal': ['Central_Junction', 'Fire_Station', 'Industrial_Zone'],
    'Fire_Station': ['West_Terminal'],
    'Industrial_Zone': ['West_Terminal'],
    'City_Hospital': ['South_Residential', 'East_Market']
}

# Graph with travel time weights used for UCS and A* algorithms
weighted_city_graph = {
    'Police_HQ': {'Traffic_Control_Center': 2, 'River_Bridge': 2},
    'Traffic_Control_Center': {'Police_HQ': 2, 'North_Station': 2},
    'North_Station': {'Traffic_Control_Center': 2, 'River_Bridge': 4, 'Central_Junction': 3},
    'River_Bridge': {'Police_HQ': 2, 'North_Station': 4}, 
    'Stadium': {'Airport_Road': 5, 'East_Market': 2},
    'Airport_Road': {'Stadium': 5, 'South_Residential': 2},
    'South_Residential': {'Airport_Road': 2, 'Central_Junction': 4, 'City_Hospital': 3},
    'East_Market': {'Stadium': 2, 'Central_Junction': 3, 'City_Hospital': 3},
    'Central_Junction': {'North_Station': 3, 'South_Residential': 4, 'East_Market': 3, 'West_Terminal': 4},
    'West_Terminal': {'Central_Junction': 4, 'Fire_Station': 2, 'Industrial_Zone': 4},
    'Fire_Station': {'West_Terminal': 2},
    'Industrial_Zone': {'West_Terminal': 4},
    'City_Hospital': {'South_Residential': 3, 'East_Market': 3}
}

# Estimated distance from each location to City Hospital for A* heuristic
heuristic = {
    "Police_HQ": 10, "Traffic_Control_Center": 8, "North_Station": 7,
    "River_Bridge": 6, "Stadium": 5, "Fire_Station": 9, "Central_Junction": 4,
    "East_Market": 3, "West_Terminal": 8, "Airport_Road": 6,
    "South_Residential": 3, "Industrial_Zone": 5, "City_Hospital": 0
}

# Module 1 Visualization Module
# This function draws the city map and highlights the recommended route
# It shows a nice picture of the path from start to destination
def visualize_city_path(standardized_request, graph_data):
    route = standardized_request.get("recommended_route")
    
    if not route or not isinstance(route, list):
        return

    G = nx.Graph()
    
    for node, edges in graph_data.items():
        if isinstance(edges, dict):
            for neighbor, weight in edges.items():
                G.add_edge(node, neighbor)
        else:
            for neighbor, weight in edges:
                G.add_edge(node, neighbor)

    pos = {
        'Police_HQ': (2, 5),
        'Traffic_Control_Center': (5, 5.5),
        'North_Station': (5, 3),
        'River_Bridge': (8, 4),
        'Stadium': (0, 2),
        'Airport_Road': (-2, 0),
        'South_Residential': (2, -1),
        'East_Market': (2, 1),
        'Central_Junction': (5, 1),
        'West_Terminal': (8, 1),
        'Fire_Station': (11, 1),
        'Industrial_Zone': (9, -2),
        'City_Hospital': (0, -3)
    }

    labels = {}
    for node in G.nodes():
        words = node.replace('_', ' ').split()
        
        if node == 'Traffic_Control_Center':
            pretty_label = 'Traffic\nControl\nCenter'
        elif node == 'South_Residential':
            pretty_label = 'South\nResidential'
        elif node == 'Central_Junction':
            pretty_label = 'Central\nJunction'
        elif node == 'West_Terminal':
            pretty_label = 'West\nTerminal'
        elif node == 'Fire_Station':
            pretty_label = 'Fire\nStation'
        elif node == 'Industrial_Zone':
            pretty_label = 'Industrial\nZone'
        elif node == 'City_Hospital':
            pretty_label = 'City\nHospital'
        elif node == 'Police_HQ':
            pretty_label = 'Police\nHQ'
        elif node == 'North_Station':
            pretty_label = 'North\nStation'
        elif node == 'River_Bridge':
            pretty_label = 'River\nBridge'
        elif node == 'Airport_Road':
            pretty_label = 'Airport\nRoad'
        elif node == 'East_Market':
            pretty_label = 'East\nMarket'
        else:
            pretty_label = words[0] if len(words) == 1 else '\n'.join(words)
        
        labels[node] = pretty_label

    path_edges = []
    for i in range(len(route) - 1):
        path_edges.append((route[i], route[i+1]))

    plt.figure(figsize=(14, 10))
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=6000, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='lightgray', width=2.0)
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight="bold", 
                           font_family="sans-serif")
    
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=4.0)

    req_id = standardized_request.get('request_id', 'Unknown')
    cat = standardized_request.get('request_category', 'Unknown')
    plt.title(f"Route Visualization: {req_id} ({cat})", fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Module 2 Data Preprocessing Module
# This module cleans and validates incoming requests
# It converts text values into numbers that the ANN can understand
def preprocess_request(raw_request):
    valid_categories = [
        "Route_Request", "Policy_Check", "Control_Allocation_Request", 
        "Emergency_Response_Request", "Integrated_City_Service_Request"
    ]
    
    category = raw_request.get("request_category", "").strip()
    if category not in valid_categories:
        raise ValueError(f"Invalid request_category: {category}")

    standardized_request = {
        "request_id": raw_request.get("request_id"),
        "request_category": category,
        "vehicle_type": raw_request.get("vehicle_type", "civilian").strip().lower(),
        "current_location": raw_request.get("current_location", "").strip(),
        "destination": raw_request.get("destination", "").strip(),
        "incident_severity": raw_request.get("incident_severity", "low").strip().lower(),
        "time_sensitivity": raw_request.get("time_sensitivity", "low").strip().lower(),
        "traffic_density": raw_request.get("traffic_density", "low").strip().lower(),
        "priority_claim": raw_request.get("priority_claim", "normal").strip().lower(),
    }

    mapping_dict = {
        "vehicle_type": {"civilian": 0, "emergency": 1},
        "levels": {"low": 0, "normal": 1, "high": 2, "critical": 3}
    }

    v_type = mapping_dict["vehicle_type"].get(standardized_request["vehicle_type"], 0)
    severity = mapping_dict["levels"].get(standardized_request["incident_severity"], 0)
    t_sens = mapping_dict["levels"].get(standardized_request["time_sensitivity"], 0)
    density = mapping_dict["levels"].get(standardized_request["traffic_density"], 0)

    standardized_request["ann_feature_vector"] = [v_type, severity, t_sens, density]

    return standardized_request

# Module 3 ANN Priority Prediction Module
# This module trains a neural network to predict request priority
# It learns from example data and assigns low/normal/high/critical levels
def train_priority_model():
    X_train = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 2],
        [1, 2, 2, 1],
        [1, 3, 3, 2],
        [1, 2, 3, 3],
        [1, 1, 1, 0]
    ])
    
    y_train = np.array([0, 1, 1, 2, 3, 3, 2])
    
    mlp = MLPClassifier(hidden_layer_sizes=(5, 4), activation='relu', max_iter=2000, random_state=42)
    mlp.fit(X_train, y_train)
    
    return mlp

# Uses trained model to predict priority level for a request
def predict_request_priority(standardized_request, trained_model):
    feature_vector = standardized_request.get("ann_feature_vector")
    
    if not feature_vector:
        raise ValueError("Missing ANN feature vector in the request.")

    prediction = trained_model.predict([feature_vector])[0]
    
    priority_mapping = {0: "low", 1: "normal", 2: "high", 3: "critical"}
    predicted_priority = priority_mapping.get(prediction, "normal")
    
    standardized_request["predicted_priority"] = predicted_priority
    
    return standardized_request

# Module 4 CSP Signal Control Module
# This module assigns traffic signal phases to intersections
# It uses MRV and forward checking to avoid conflicts between nearby signals
def assign_control_signals(standardized_request):
    csp_variables = ['North_Station', 'Central_Junction', 'East_Market', 'City_Hospital']

    csp_conflict_graph = {
        'North_Station': ['Central_Junction'],
        'Central_Junction': ['North_Station', 'East_Market'],
        'East_Market': ['Central_Junction', 'City_Hospital'],
        'City_Hospital': ['East_Market']
    }

    initial_domains = {
        'North_Station': ['PhaseA', 'PhaseB', 'PhaseC'],
        'Central_Junction': ['PhaseA', 'PhaseB'],
        'East_Market': ['PhaseB', 'PhaseC'],
        'City_Hospital': ['PhaseA', 'PhaseC']
    }

    # Selects the variable with fewest remaining values
    def select_unassigned_variable_mrv(assignment, domains, variables):
        min_length = 999999
        best_variable = None
        for var in variables:
            if var not in assignment:
                current_domain_length = len(domains[var])
                if current_domain_length < min_length:
                    min_length = current_domain_length
                    best_variable = var
        return best_variable

    # Removes conflicting values from neighbor domains
    def forward_check(assigned_var, assigned_value, assignment, domains, graph):
        for neighbor in graph[assigned_var]:
            if neighbor in domains and neighbor not in assignment:
                if assigned_value in domains[neighbor]:
                    domains[neighbor].remove(assigned_value)
                if len(domains[neighbor]) == 0:
                    return False 
        return True

    # Recursively tries to assign values to all variables
    def backtrack_search(assignment, domains, variables, graph):
        if len(assignment) == len(variables):
            return assignment

        current_var = select_unassigned_variable_mrv(assignment, domains, variables)
        available_values = list(domains[current_var])

        for value in available_values:
            assignment[current_var] = value
            
            if forward_check(current_var, value, assignment, domains, graph):
                result = backtrack_search(assignment, domains, variables, graph)
                if result is not None:
                    return result

            del assignment[current_var]

        return None

    assignment = {}
    solution = backtrack_search(assignment, initial_domains, csp_variables, csp_conflict_graph)
    
    if solution:
        standardized_request["assigned_control_plan"] = solution
    else:
        standardized_request["assigned_control_plan"] = "Failed to find safe allocation"
        
    return standardized_request

# Search Algorithms Helper Functions
# Builds path from parent dictionary after search completes
def constructpath(parent, start, goal):
    if goal not in parent:
        return None
    path = [goal]
    prev = goal
    while parent[prev] is not None:
        path.insert(0, parent[prev])
        prev = parent[prev]
    return path

# Breadth First Search finds shortest path in unweighted graph
def BFS(graph, start, goal):
    visited = set()
    parent = {start: None}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node == goal:
            return constructpath(parent, start, goal)

        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited and neighbor not in parent:
                    queue.append(neighbor)
                    parent[neighbor] = node
    return None

# A* search algorithm uses heuristic to find optimal path faster
def A_staric(graph, start, goal):
    visited = set()
    parent = {start: None}
    PQ = []

    heapq.heappush(PQ, (heuristic[start], 0, start))

    while PQ:
        f, g, node = heapq.heappop(PQ)

        if node == goal:
            return constructpath(parent, start, goal), g

        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph[node].items():
                if neighbor not in visited:
                    new_g = g + weight
                    new_f = new_g + heuristic[neighbor]
                    heapq.heappush(PQ, (new_f, new_g, neighbor))
                    parent[neighbor] = node

    return None, float("inf")

# Uniform Cost Search finds cheapest path by travel time
def ucs(graph, start, goal):
    pq = []
    path = [start]
    heapq.heappush(pq, (0, start, path))
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        
        if node == goal:
            return cost, path
         
        if node in visited:
            continue
            
        visited.add(node)
        
        for neighbor, weight in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
                
    return float("inf"), None

# Module 5 Search and Navigation Module
# This module finds the best route from start to destination
# It uses BFS for simple routes, UCS for normal requests, and A* for emergencies
def search_and_navigation_module(standardized_request):
    start_node = standardized_request.get("current_location")
    goal_node = standardized_request.get("destination")
    category = standardized_request.get("request_category")

    if category == "Route_Request":
        path = BFS(unweighted_city_graph, start_node, goal_node)
        standardized_request["recommended_route"] = path
        standardized_request["travel_info"] = f"{len(path)-1} segments" if path else "No path found"

    elif category in ["Emergency_Response_Request", "Integrated_City_Service_Request"]:
        path, cost = A_staric(weighted_city_graph, start_node, goal_node)
        standardized_request["recommended_route"] = path
        standardized_request["travel_info"] = f"{cost} Mins (A*)"

    else:
        cost, path = ucs(weighted_city_graph, start_node, goal_node)
        standardized_request["recommended_route"] = path
        standardized_request["travel_info"] = f"{cost} Mins (UCS)"

    return standardized_request

# Module 6 First Order Logic Validation Module
# This module checks if the request follows city traffic policies
# It uses logical rules to approve or reject the request
def validate_request_logic(standardized_request):
    kb = FolKB()

    kb.tell(expr("EmergencyVehicle(v) & IncidentSeverity(v,High) ==> Priority(v,Critical)"))
    kb.tell(expr("EmergencyVehicle(v) & TimeSensitive(v) ==> Priority(v,High)"))
    kb.tell(expr("CivilianVehicle(v) ==> Priority(v,Normal)"))

    kb.tell(expr("EmergencyVehicle(v) & SignalZone(z) ==> Authorized(v,SignalOverride)"))
    kb.tell(expr("CivilianVehicle(v) & SignalZone(z) ==> Unauthorized(v,SignalOverride)"))

    kb.tell(expr("EmergencyVehicle(v) & Destination(v,h) & Hospital(h) ==> EmergencyCorridor(v)"))
    kb.tell(expr("EmergencyCorridor(v) ==> Authorized(v,EmergencyRoute)"))

    kb.tell(expr("Authorized(v,action) ==> AllowedAction(v,action)"))
    kb.tell(expr("Unauthorized(v,action) ==> DisallowedAction(v,action)"))

    kb.tell(expr("Priority(v,Critical) & Unauthorized(v,action) ==> DisallowedAction(v,action)"))
    kb.tell(expr("Priority(v,Critical) & Authorized(v,EmergencyRoute) ==> AllowedAction(v,SignalOverride)"))

    kb.tell(expr("AllowedAction(v,action) ==> Approved(v,req)"))
    kb.tell(expr("DisallowedAction(v,action) ==> Rejected(v,req)"))

    kb.tell(expr("RequestType(req,Route_Request) ==> Approved(v,req)"))
    kb.tell(expr("RequestType(req,Policy_Check) & Authorized(v,action) ==> Approved(v,req)"))
    kb.tell(expr("RequestType(req,Policy_Check) & Unauthorized(v,action) ==> Rejected(v,req)"))
    kb.tell(expr("RequestType(req,Control_Allocation_Request) & AllowedAction(v,action) ==> Approved(v,req)"))
    kb.tell(expr("RequestType(req,Control_Allocation_Request) & DisallowedAction(v,action) ==> Rejected(v,req)"))
    kb.tell(expr("RequestType(req,Emergency_Response_Request) & Priority(v,level) & Authorized(v,EmergencyRoute) ==> Approved(v,req)"))
    kb.tell(expr("RequestType(req,Integrated_City_Service_Request) & Priority(v,Critical) & Authorized(v,EmergencyRoute) & AllowedAction(v,action) ==> Approved(v,req)"))

    v_const = "Veh1"
    r_const = "Req1"

    req_category = standardized_request.get("request_category", "Unknown")
    kb.tell(expr(f"RequestType({r_const},{req_category})"))

    if standardized_request.get("vehicle_type", "civilian") == "emergency":
        kb.tell(expr(f"EmergencyVehicle({v_const})"))
    else:
        kb.tell(expr(f"CivilianVehicle({v_const})"))

    severity = standardized_request.get("incident_severity", "")
    if severity in ["high", "critical"]:
        kb.tell(expr(f"IncidentSeverity({v_const},High)"))

    time_sens = standardized_request.get("time_sensitivity", "")
    if time_sens in ["high", "critical"]:
        kb.tell(expr(f"TimeSensitive({v_const})"))

    if standardized_request.get("destination", "").lower() == "city_hospital":
        kb.tell(expr(f"Destination({v_const},CityHospital)"))
        kb.tell(expr("Hospital(CityHospital)"))

    kb.tell(expr("SignalZone(Zone1)"))

    is_approved = kb.ask(expr(f"Approved({v_const},{r_const})"))
    is_rejected = kb.ask(expr(f"Rejected({v_const},{r_const})"))

    if is_approved is not False:
        standardized_request["policy_validation"] = "Approved"
    elif is_rejected is not False:
        standardized_request["policy_validation"] = "Rejected"
    else:
        standardized_request["policy_validation"] = "Pending_Manual_Review"

    return standardized_request

# Module 7 Request Router Module
# This module directs the request to the right AI modules
# It checks the category and calls the appropriate functions
def request_router(standardized_request, trained_ann_model=None):
    category = standardized_request.get("request_category")

    if category == "Route_Request":
        standardized_request = search_and_navigation_module(standardized_request)

    elif category == "Policy_Check":
        standardized_request = validate_request_logic(standardized_request)

    elif category == "Control_Allocation_Request":
        standardized_request = validate_request_logic(standardized_request)
        
        if standardized_request.get("policy_validation") == "Approved":
            standardized_request = assign_control_signals(standardized_request)
        else:
            standardized_request["assigned_control_plan"] = "Skipped - Action not authorized"

    elif category in ["Emergency_Response_Request", "Integrated_City_Service_Request"]:
        
        if trained_ann_model is not None:
            standardized_request = predict_request_priority(standardized_request, trained_ann_model)
        else:
            standardized_request["predicted_priority"] = "System Error: No ANN Model loaded"

        standardized_request = validate_request_logic(standardized_request)

        if standardized_request.get("policy_validation") == "Approved":
            standardized_request = assign_control_signals(standardized_request)
            standardized_request = search_and_navigation_module(standardized_request)
        else:
            standardized_request["assigned_control_plan"] = "Skipped - Action not authorized"
            standardized_request["recommended_route"] = "Skipped - Action not authorized"
            standardized_request["travel_info"] = "N/A"

    else:
        raise ValueError(f"Unknown Request Category: {category}")

    return standardized_request

# Module 8 Final Response Module
# This module collects all outputs and creates a clean response
# It shows the decision, route, control plan, and priority level
def generate_final_response(standardized_request):
    final_output = {
        "Request ID": standardized_request.get("request_id", "Unknown"),
        "Category": standardized_request.get("request_category", "Unknown"),
        "Vehicle": standardized_request.get("vehicle_type", "civilian").capitalize()
    }

    if "predicted_priority" in standardized_request:
        final_output["Predicted Priority"] = standardized_request["predicted_priority"].upper()

    if "policy_validation" in standardized_request:
        final_output["Policy Status"] = standardized_request["policy_validation"]

    if "assigned_control_plan" in standardized_request:
        plan = standardized_request["assigned_control_plan"]
        if isinstance(plan, dict):
            formatted_plan = ", ".join([f"{node}: {phase}" for node, phase in plan.items()])
            final_output["Assigned Control Plan"] = formatted_plan
        else:
            final_output["Assigned Control Plan"] = plan

    if "recommended_route" in standardized_request:
        route = standardized_request["recommended_route"]
        if isinstance(route, list):
            final_output["Recommended Route"] = " -> ".join(route)
        else:
             final_output["Recommended Route"] = route
             
        final_output["Travel Estimate"] = standardized_request.get("travel_info", "N/A")

    status = final_output.get("Policy Status", "Approved")
    
    if status == "Rejected":
        message = f"Request REJECTED. The requested action is unauthorized under current city policy."
    else:
        parts = ["Request APPROVED."]
        if "Predicted Priority" in final_output:
            parts.append(f"Operating under {final_output['Predicted Priority']} priority.")
        if isinstance(standardized_request.get("assigned_control_plan"), dict):
            parts.append("Emergency corridor and signal overrides have been secured.")
        if "Recommended Route" in final_output:
            parts.append("Navigation path is ready.")
        message = " ".join(parts)
        
    final_output["Decision Message"] = message

    print("\n")
    print(f" FINAL RESPONSE: {final_output['Request ID']}")
    print("\n")
    for key, value in final_output.items():
        if key == "Decision Message":
            print("\n")
        print(f"{key+':':<22} {value}")
    print("\n")

    return final_output

# Sample request 1 Simple route request from Stadium to South Residential
request_1_route = {
    "request_id": "REQ-001",
    "request_category": "Route_Request",
    "vehicle_type": "civilian",
    "current_location": "Stadium", 
    "destination": "South_Residential",
    "incident_severity": "low",
    "time_sensitivity": "low",
    "traffic_density": "normal",
    "priority_claim": "normal"
}

# Sample request 2 Policy check for emergency vehicle going to hospital
request_2_policy = {
    "request_id": "REQ-002",
    "request_category": "Policy_Check",
    "vehicle_type": "emergency",
    "current_location": "Central_Junction",
    "destination": "City_Hospital",
    "incident_severity": "high",
    "time_sensitivity": "high",
    "traffic_density": "high",
    "priority_claim": "high"
}

# Sample request 3 Control allocation request for emergency vehicle
request_3_control = {
    "request_id": "REQ-003",
    "request_category": "Control_Allocation_Request",
    "vehicle_type": "emergency",
    "current_location": "North_Station",
    "destination": "City_Hospital",
    "incident_severity": "critical",
    "time_sensitivity": "high",
    "traffic_density": "normal",
    "priority_claim": "high"
}

# Sample request 4 Full emergency response request
request_4_emergency = {
    "request_id": "REQ-004",
    "request_category": "Emergency_Response_Request",
    "vehicle_type": "emergency",
    "current_location": "North_Station",
    "destination": "City_Hospital",
    "incident_severity": "critical",
    "time_sensitivity": "critical",
    "traffic_density": "high",
    "priority_claim": "critical"
}

# Train the ANN model first
trained_mlp = train_priority_model()

# List of all test requests
all_requests = [
    request_1_route, 
    request_2_policy, 
    request_3_control, 
    request_4_emergency
]

# Process each request and show results with visualization
for raw_req in all_requests:
    std_req = preprocess_request(raw_req)
    processed_req = request_router(std_req, trained_mlp)
    generate_final_response(processed_req)
    visualize_city_path(processed_req, weighted_city_graph)
