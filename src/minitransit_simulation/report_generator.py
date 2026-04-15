import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# ==========================================
# 1. Directory Setup & Path Resolutions
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent 

# Input and Output paths based on your requirements
INPUT_DIR = SRC_DIR / "data" / "simulation_results"
OUTPUT_DIR = SRC_DIR / "data" / "simulation_reports"

def parse_time_to_dt(time_str):
    """
    Helper to handle "08:00" format.
    Necessary for time delta math and bucketizing.
    """
    if not time_str or ':' not in time_str:
        return None
    try:
        # Assuming a base date for calculation purposes
        return datetime.strptime(time_str, "%H:%M")
    except Exception:
        return None

def get_time_bucket(dt, interval_minutes=15):
    """
    Floors a datetime into a 15-min bucket (e.g., 08:14 -> 08:00).
    """
    if not dt:
        return "Unknown"
    minute = (dt.minute // interval_minutes) * interval_minutes
    return dt.replace(minute=minute, second=0).strftime("%H:%M")

def generate_reports(json_filename):
    if not json_filename.endswith('.json'):
        json_filename += '.json'
        
    input_filepath = INPUT_DIR / json_filename
    
    if not input_filepath.exists():
        print(f"[Error] File not found: {input_filepath}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_prefix = input_filepath.stem
    
    print(f"[*] Parsing JSON data: {json_filename}...")
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Standard format: data is a dict with a "routes" key
    routes = data.get('routes', [])
    
    if not routes:
        print("[Warning] No valid routes found in the JSON file.")
        return

    # ==========================================
    # 2. Data Containers for the 8 Dimensions
    # ==========================================
    global_metrics = {
        "total_passengers": 0,
        "total_travel_time_sec": 0,
        "total_wait_time_sec": 0,
        "total_walk_time_sec": 0,
        "total_ride_time_sec": 0, 
        "total_transfers": 0
    }
    modal_split = defaultdict(int)
    
    # Hex Level Stats (Spatial & Equity)
    spatial_stats = defaultdict(lambda: {
        "boardings": 0, "alightings": 0, "total_wait_sec": 0, "wait_count": 0,
        "total_walk_origin_sec": 0, "walk_origin_count": 0
    })
    
    # Fleet/Service Stats (Operations & Resource Utilization)
    fleet_stats = defaultdict(lambda: {
        "trips": 0, "passengers_carried": 0, "service_name": "Unknown", "active_drive_time_sec": 0
    })

    # Temporal Stats (Dynamics)
    temporal_stats = defaultdict(lambda: {
        "departures": 0, "arrivals": 0, "total_wait_sec": 0, "wait_count": 0
    })

    # OD Corridor Stats
    od_pairs = defaultdict(lambda: {"count": 0, "total_travel_sec": 0})

    # ==========================================
    # 3. Core Processing Logic
    # ==========================================
    for route in routes:
        unit = route.get("unit", 0)
        global_metrics["total_passengers"] += unit
        
        actions = route.get("actions", [])
        if not actions:
            continue
            
        # 3.1 Global Travel Time & OD Analysis
        # 1.json provides duration_minutes per action
        total_route_min = sum(a.get("duration_minutes", 0) for a in actions)
        travel_time_sec = total_route_min * 60
        global_metrics["total_travel_time_sec"] += travel_time_sec * unit
        
        start_time_dt = parse_time_to_dt(actions[0].get("start_time"))
        end_time_dt = parse_time_to_dt(actions[-1].get("end_time"))
        
        start_hex = actions[0].get("start_hex", "Unknown")
        end_hex = actions[-1].get("end_hex", "Unknown")
        od_key = f"{start_hex} -> {end_hex}"
        od_pairs[od_key]["count"] += unit
        od_pairs[od_key]["total_travel_sec"] += travel_time_sec * unit

        # Temporal Buckets
        if start_time_dt:
            temporal_stats[get_time_bucket(start_time_dt)]["departures"] += unit
        if end_time_dt:
            temporal_stats[get_time_bucket(end_time_dt)]["arrivals"] += unit

        # 3.2 Modal Split & Transfers
        ride_actions = [a for a in actions if a.get("type") in ["Ride", "OnDemandRide"]]
        global_metrics["total_transfers"] += max(0, len(ride_actions) - 1) * unit

        has_walk = False
        has_ride = False
        has_ondemand = False

        for action in actions:
            act_type = action.get("type", "")
            if act_type == "Walk":
                has_walk = True
            elif act_type == "Ride":
                has_ride = True
            elif act_type == "OnDemandRide":
                has_ondemand = True

        if has_ride and has_ondemand:
            modal_split["Bike + Bus"] += unit
        elif has_ride:
            modal_split["Walk + Bus"] += unit
        elif has_ondemand:
            modal_split["Walk + Bike"] += unit
        elif has_walk:
            modal_split["Walk Only"] += unit
        else:
            modal_split["Other"] += unit

        # 3.3 Micro-Action Analysis
        for action in actions:
            a_type = action.get("type")
            a_dur_sec = action.get("duration_minutes", 0) * 60
            a_start_hex = action.get("start_hex")
            a_end_hex = action.get("end_hex")
            a_start_time = parse_time_to_dt(action.get("start_time"))

            if a_type == "Walk":
                global_metrics["total_walk_time_sec"] += a_dur_sec * unit
                if a_start_hex:
                    spatial_stats[a_start_hex]["total_walk_origin_sec"] += a_dur_sec * unit
                    spatial_stats[a_start_hex]["walk_origin_count"] += unit

            elif a_type == "Wait":
                global_metrics["total_wait_time_sec"] += a_dur_sec * unit
                if a_start_hex:
                    spatial_stats[a_start_hex]["total_wait_sec"] += a_dur_sec * unit
                    spatial_stats[a_start_hex]["wait_count"] += unit
                if a_start_time:
                    bucket = get_time_bucket(a_start_time)
                    temporal_stats[bucket]["total_wait_sec"] += a_dur_sec * unit
                    temporal_stats[bucket]["wait_count"] += unit
                    
            elif a_type in ["Ride", "OnDemandRide"]:
                global_metrics["total_ride_time_sec"] += a_dur_sec * unit
                if a_start_hex: spatial_stats[a_start_hex]["boardings"] += unit
                if a_end_hex: spatial_stats[a_end_hex]["alightings"] += unit
                
                # Service identifier
                svc = action.get("service_name") or action.get("vehicle") or "Unknown_Service"
                fleet_stats[svc]["trips"] += 1
                fleet_stats[svc]["passengers_carried"] += unit
                fleet_stats[svc]["service_name"] = svc
                fleet_stats[svc]["active_drive_time_sec"] += a_dur_sec

    # ==========================================
    # 4. Generate the 5 CSV Reports
    # ==========================================
    total_pax = global_metrics["total_passengers"]
    
    # Report 1: Overview (Macro, Micro, Sustainability)
    with open(OUTPUT_DIR / f"{file_prefix}_overview.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Metric", "Value"])
        writer.writerow(["Macro", "Total Passengers", total_pax])
        writer.writerow(["Macro", "Avg Transfers", round(global_metrics["total_transfers"] / total_pax, 2) if total_pax else 0])
        writer.writerow(["Modal Split", "Walk Only", modal_split["Walk Only"]])
        writer.writerow(["Modal Split", "Walk + Bus", modal_split["Walk + Bus"]])
        writer.writerow(["Modal Split", "Walk + Bike", modal_split["Walk + Bike"]])
        writer.writerow(["Modal Split", "Bike + Bus", modal_split["Bike + Bus"]])
        writer.writerow(["Micro Experience", "Avg Travel Time (min)", round((global_metrics["total_travel_time_sec"] / total_pax) / 60, 2) if total_pax else 0])
        writer.writerow(["Micro Experience", "Avg Wait Time (min)", round((global_metrics["total_wait_time_sec"] / total_pax) / 60, 2) if total_pax else 0])
        writer.writerow(["Micro Experience", "Avg Walk Time (min)", round((global_metrics["total_walk_time_sec"] / total_pax) / 60, 2) if total_pax else 0])
        writer.writerow(["Sustainability", "Total Vehicle Service Time (hrs)", round(global_metrics["total_ride_time_sec"] / 3600, 2)])

    # Report 2: Spatial & Equity (Hex Level)
    with open(OUTPUT_DIR / f"{file_prefix}_spatial_hex.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Hex_ID", "Boardings", "Alightings", "Avg_Wait_min", "Avg_Walk_Burden_min"])
        for h, s in spatial_stats.items():
            avg_wait = round((s["total_wait_sec"] / s["wait_count"]) / 60, 2) if s["wait_count"] else 0
            avg_walk = round((s["total_walk_origin_sec"] / s["walk_origin_count"]) / 60, 2) if s["walk_origin_count"] else 0
            writer.writerow([h, s["boardings"], s["alightings"], avg_wait, avg_walk])

    # Report 3: Fleet Operations (Resource Utilization)
    with open(OUTPUT_DIR / f"{file_prefix}_fleet_operations.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Service_Name", "Total_Trips", "Total_Passengers", "Active_Drive_hrs"])
        sorted_fleet = sorted(fleet_stats.items(), key=lambda x: x[1]["passengers_carried"], reverse=True)
        for svc_id, stats in sorted_fleet:
            writer.writerow([svc_id, stats["trips"], stats["passengers_carried"], round(stats["active_drive_time_sec"] / 3600, 2)])

    # Report 4: Temporal Dynamics (15-min intervals)
    with open(OUTPUT_DIR / f"{file_prefix}_temporal_dynamics.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Bucket", "Departures", "Arrivals", "Avg_Wait_min"])
        for bucket in sorted(temporal_stats.keys()):
            s = temporal_stats[bucket]
            avg_wait = round((s["total_wait_sec"] / s["wait_count"]) / 60, 2) if s["wait_count"] else 0
            writer.writerow([bucket, s["departures"], s["arrivals"], avg_wait])

    # Report 5: Top OD Pairs (Corridors)
    with open(OUTPUT_DIR / f"{file_prefix}_top_od_pairs.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["OD_Pair", "Total_Passengers", "Avg_Travel_Time_min"])
        sorted_od = sorted(od_pairs.items(), key=lambda x: x[1]["count"], reverse=True)
        for od_key, stats in sorted_od:
            avg_time = round((stats["total_travel_sec"] / stats["count"]) / 60, 2)
            writer.writerow([od_key, stats["count"], avg_time])

    print(f"[Success] Comprehensive reports generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Full-Dimension Transport Reports")
    parser.add_argument("json_file", type=str, help="Input JSON filename")
    args = parser.parse_args()
    generate_reports(args.json_file)