# report_generator.py — User Guide

## Purpose

`report_generator.py` converts simulation output from JSON into user-friendly CSV reports.

The simulation output is originally a JSON file containing travel routes and action sequences. This script analyzes that JSON and generates summary reports so users do not need to parse raw JSON manually.

---

## What It Does

The script reads a JSON file from:

- `src/data/simulation_results`

It then computes and exports report files into:

- `src/data/simulation_reports`

It summarizes the simulation from multiple perspectives:

- overall passenger and travel metrics
- modal split
- spatial equity per hex cell
- fleet and service utilization
- temporal dynamics in 15-minute intervals
- top OD corridor travel patterns

---

## Generated Output Files

The script produces 5 CSV reports, all sharing the same input file prefix:

1. `*_overview.csv`

   - Total passengers
   - Average transfers
   - Modal split: Walk Only / Walk + Bus / Walk + Bike / Bike + Bus
   - Average travel, wait, and walk times
   - Total vehicle service hours
2. `*_spatial_hex.csv`

   - Hex ID
   - Boardings
   - Alightings
   - Average wait time
   - Average walk burden time
3. `*_fleet_operations.csv`

   - Service or vehicle name
   - Total trips
   - Total passengers carried
   - Active driving hours
4. `*_temporal_dynamics.csv`

   - 15-minute time bucket
   - Departures
   - Arrivals
   - Average wait time
5. `*_top_od_pairs.csv`

   - OD pair (`start_hex -> end_hex`)
   - Total passengers
   - Average travel time

---

## Input Requirements

The input JSON must contain a top-level `routes` array.

Each route can include:

- `unit` (passenger count)
- `actions` list

Each action may include:

- `type` (`Walk`, `Wait`, `Ride`, `OnDemandRide`)
- `duration_minutes`
- `start_time`, `end_time`
- `start_hex`, `end_hex`
- `service_name` or `vehicle`

---

## How to Run

From the script directory:

```bash
python report_generator.py your_simulation_result.json
```

---

## Notes

* The output directory is created automatically if it does not exist.
* If the input file is missing, the script prints an error.
* If the JSON contains no routes, the script prints a warning.

---

## Why Use This Script

This file is designed for users who want:

* easier analysis without handling raw JSON
* quick CSV exports for Excel or reporting
* multi-dimensional transport insights from simulation results
