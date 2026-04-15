"""
Delhi Metro RFI Data Generator
Generates 500 synthetic Request for Inspection records with realistic patterns.
Ensures CTR-01 at STN-08-Kirti-Nagar has ~62% rejection rate.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# --- Configuration ---
NUM_RECORDS = 500
CONTEXT_DATE = datetime(2026, 4, 15)

PACKAGES = ["PKG-CIVIL", "PKG-ELEC", "PKG-MECH", "PKG-SIGNAL", "PKG-TRACK"]

STATIONS = [
    "STN-01-Janakpuri-West", "STN-02-Janakpuri-East", "STN-03-Tilak-Nagar",
    "STN-04-Subhash-Nagar", "STN-05-Rajouri-Garden", "STN-06-Ramesh-Nagar",
    "STN-07-Moti-Nagar", "STN-08-Kirti-Nagar", "STN-09-Shadipur",
    "STN-10-Patel-Nagar", "STN-11-Rajendra-Place", "STN-12-Karol-Bagh"
]

SUBSYSTEMS = [
    "Structural", "Electrical", "HVAC", "Plumbing", "Fire-Safety",
    "Signaling", "Telecom", "Track-Work", "Finishing", "Waterproofing"
]

ACTIVITIES = [
    "Rebar Inspection", "Concrete Pour Check", "Weld Joint Test",
    "Alignment Survey", "Insulation Check", "Cable Tray Inspection",
    "Fire Alarm Test", "Waterproofing Membrane Check", "Track Gauge Verify",
    "Tile Laying Inspection", "HVAC Duct Install Check", "Earthing Test",
    "Paint Coat Inspection", "Bolt Torque Verification", "Drainage Slope Check",
    "Signal Relay Test", "Platform Edge Verify", "Ventilation CFM Test",
    "Emergency Exit Sign Check", "Column Verticality Check"
]

INITIATOR_ROLES = ["Site-Engineer", "QA-Manager", "Project-Manager", "Supervisor"]

CONTRACTORS = ["CTR-01", "CTR-02", "CTR-03", "CTR-04", "CTR-05",
               "CTR-06", "CTR-07", "CTR-08"]

INSPECTORS = ["INS-01", "INS-02", "INS-03", "INS-04", "INS-05",
              "INS-06", "INS-07", "INS-08", "INS-09", "INS-10"]

QUANTITY_UNITS = ["m", "m²", "m³", "kg", "nos", "liters", "sets"]

STATUSES = ["Approved", "Rejected", "Open", "Conditionally-Approved"]

REMARKS_TEMPLATES = {
    "Approved": [
        "Work completed as per specification. No defects observed.",
        "Inspection satisfactory. Quality meets standards.",
        "All parameters within acceptable limits. Approved for next stage.",
        "Good workmanship. Approved.",
        "Material test reports verified. Work accepted.",
        "Alignment within tolerance. Approved.",
    ],
    "Rejected": [
        "Surface cracks observed. Rework required before re-inspection.",
        "Alignment deviation exceeds tolerance. Needs correction.",
        "Material quality below specification. Replace and re-submit.",
        "Improper curing observed. Concrete strength insufficient.",
        "Weld defect detected via NDT. Grinding and re-welding required.",
        "Inadequate cover to reinforcement. Rectify immediately.",
        "Waterproofing membrane has punctures. Patch and re-inspect.",
        "Cable insulation resistance below threshold. Replace cables.",
        "Fire safety clearance gap found. Seal all penetrations.",
        "Surface finishing rough. Sand and re-coat required.",
        "Bolt torque values below specification. Re-torque all bolts.",
        "Drainage slope insufficient. Re-grade and re-inspect.",
        "Paint coat thickness below minimum. Apply additional coat.",
        "Earthing resistance too high. Improve earthing pit.",
    ],
    "Open": [
        "Inspection scheduled. Awaiting site readiness.",
        "Pending material test report submission.",
        "Waiting for contractor to complete rework.",
        "Deferred due to site access issues.",
        "Re-inspection pending after previous rejection.",
    ],
    "Conditionally-Approved": [
        "Minor punch items noted. Complete within 7 days.",
        "Approved with condition to submit revised drawing.",
        "Accepted subject to additional protective coating.",
        "Conditional approval. Rectify snag list within 2 weeks.",
        "Approved conditionally. Submit lab test results by next week.",
    ],
}


def generate_records():
    records = []
    rfi_counter = 1

    # --- Phase 1: Generate ~50 records for CTR-01 at STN-08-Kirti-Nagar ---
    # Target: 62% rejection rate => ~31 rejected, ~19 approved/other, ~1-2 open
    ctr01_stn08_count = 50
    ctr01_stn08_rejected = 31
    ctr01_stn08_approved = 15
    ctr01_stn08_cond = 2
    ctr01_stn08_open = 2

    statuses_block = (
        ["Rejected"] * ctr01_stn08_rejected +
        ["Approved"] * ctr01_stn08_approved +
        ["Conditionally-Approved"] * ctr01_stn08_cond +
        ["Open"] * ctr01_stn08_open
    )
    random.shuffle(statuses_block)

    # Some activities should repeat 3+ times with rejection at this station
    repeat_activities = random.sample(ACTIVITIES, 5)

    for i in range(ctr01_stn08_count):
        status = statuses_block[i]
        # Use repeat activities for first ~25 records to ensure Rule 2 triggers
        if i < 25:
            activity = repeat_activities[i % 5]
        else:
            activity = random.choice(ACTIVITIES)

        raised = CONTEXT_DATE - timedelta(days=random.randint(10, 180))
        sla_days = random.randint(5, 21)
        sla_deadline = raised + timedelta(days=sla_days)

        if status == "Open":
            closed = None
        elif status == "Rejected":
            # Some close prematurely (before SLA), some after
            close_offset = random.randint(1, sla_days + 10)
            closed = raised + timedelta(days=close_offset)
        else:
            close_offset = random.randint(3, sla_days + 5)
            closed = raised + timedelta(days=close_offset)

        planned_qty = round(random.uniform(10, 500), 1)
        if status in ["Approved", "Conditionally-Approved"]:
            verified_qty = round(planned_qty * random.uniform(0.85, 1.0), 1)
        elif status == "Rejected":
            verified_qty = round(planned_qty * random.uniform(0.0, 0.5), 1)
        else:
            verified_qty = 0.0

        records.append({
            "rfi_id": f"RFI-{rfi_counter:04d}",
            "package": random.choice(PACKAGES),
            "station": "STN-08-Kirti-Nagar",
            "subsystem_type": random.choice(SUBSYSTEMS),
            "activity_name": activity,
            "initiator_role": random.choice(INITIATOR_ROLES),
            "contractor_id": "CTR-01",
            "inspector_id": random.choice(INSPECTORS),
            "raised_date": raised.strftime("%Y-%m-%d"),
            "sla_deadline": sla_deadline.strftime("%Y-%m-%d"),
            "closed_date": closed.strftime("%Y-%m-%d") if closed else "",
            "inspection_status": status,
            "planned_quantity": planned_qty,
            "verified_quantity": verified_qty,
            "quantity_unit": random.choice(QUANTITY_UNITS),
            "remarks": random.choice(REMARKS_TEMPLATES[status]),
        })
        rfi_counter += 1

    # --- Phase 2: Generate remaining records (450) ---
    total_open_target = 20  # 22 total open (2 from block above)
    open_count = 0

    for _ in range(NUM_RECORDS - ctr01_stn08_count):
        station = random.choice(STATIONS)
        contractor = random.choice(CONTRACTORS)

        # Avoid duplicating the hidden pattern combination outside the block
        if contractor == "CTR-01" and station == "STN-08-Kirti-Nagar":
            contractor = random.choice(["CTR-02", "CTR-03", "CTR-04"])

        if open_count < total_open_target and random.random() < 0.06:
            status = "Open"
            open_count += 1
        else:
            status = random.choices(
                ["Approved", "Rejected", "Conditionally-Approved"],
                weights=[55, 25, 20],
                k=1
            )[0]

        activity = random.choice(ACTIVITIES)
        raised = CONTEXT_DATE - timedelta(days=random.randint(5, 300))
        sla_days = random.randint(5, 21)
        sla_deadline = raised + timedelta(days=sla_days)

        if status == "Open":
            closed = None
        else:
            close_offset = random.randint(2, sla_days + 8)
            closed = raised + timedelta(days=close_offset)

        planned_qty = round(random.uniform(10, 500), 1)
        if status in ["Approved", "Conditionally-Approved"]:
            verified_qty = round(planned_qty * random.uniform(0.85, 1.0), 1)
        elif status == "Rejected":
            verified_qty = round(planned_qty * random.uniform(0.0, 0.5), 1)
        else:
            verified_qty = 0.0

        records.append({
            "rfi_id": f"RFI-{rfi_counter:04d}",
            "package": random.choice(PACKAGES),
            "station": station,
            "subsystem_type": random.choice(SUBSYSTEMS),
            "activity_name": activity,
            "initiator_role": random.choice(INITIATOR_ROLES),
            "contractor_id": contractor,
            "inspector_id": random.choice(INSPECTORS),
            "raised_date": raised.strftime("%Y-%m-%d"),
            "sla_deadline": sla_deadline.strftime("%Y-%m-%d"),
            "closed_date": closed.strftime("%Y-%m-%d") if closed else "",
            "inspection_status": status,
            "planned_quantity": planned_qty,
            "verified_quantity": verified_qty,
            "quantity_unit": random.choice(QUANTITY_UNITS),
            "remarks": random.choice(REMARKS_TEMPLATES[status]),
        })
        rfi_counter += 1

    # Ensure we hit exactly 22 open if needed
    # Force remaining open slots
    remaining_open = total_open_target - open_count
    if remaining_open > 0:
        # Convert some of the last non-CTR01/STN08 approved records to open
        for idx in range(len(records) - 1, ctr01_stn08_count, -1):
            if remaining_open <= 0:
                break
            if records[idx]["inspection_status"] == "Approved" and records[idx]["station"] != "STN-08-Kirti-Nagar":
                records[idx]["inspection_status"] = "Open"
                records[idx]["closed_date"] = ""
                records[idx]["verified_quantity"] = 0.0
                records[idx]["remarks"] = random.choice(REMARKS_TEMPLATES["Open"])
                remaining_open -= 1

    random.shuffle(records)
    return records


if __name__ == "__main__":
    data = generate_records()
    df = pd.DataFrame(data)

    # Verify hidden pattern
    mask = (df["contractor_id"] == "CTR-01") & (df["station"] == "STN-08-Kirti-Nagar")
    subset = df[mask]
    rejection_rate = (subset["inspection_status"] == "Rejected").sum() / len(subset) * 100
    open_count = (df["inspection_status"] == "Open").sum()

    print(f"Total records: {len(df)}")
    print(f"CTR-01 @ STN-08-Kirti-Nagar: {len(subset)} records, {rejection_rate:.1f}% rejection rate")
    print(f"Total Open RFIs: {open_count}")

    df.to_csv("delhi_metro_rfi_data.csv", index=False)
    print("✅ CSV saved as delhi_metro_rfi_data.csv")
