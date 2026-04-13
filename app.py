from flask import Flask, request, render_template, redirect, url_for, session
from flask import jsonify
from src.planner import generate_schedule, format_time
import json
import csv
import copy
import os
from pathlib import Path
from datetime import datetime, date

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "secret123")


@app.context_processor
def inject_static_version():
    style_path = Path(app.root_path) / "static" / "style.css"
    try:
        version = int(style_path.stat().st_mtime)
    except OSError:
        version = 1
    return {"static_version": version}

POMODORO_TIERS = {
    "light": (25, 5),
    "standard": (50, 10),
    "deep": (90, 15)
}


# ---------- FILE HELPERS ----------
def load_users():
    try:
        with open("data/users.json") as f:
            users = json.load(f)
            return normalize_users(users)
    except:
        return []

def save_users(data):
    with open("data/users.json", "w") as f:
        json.dump(normalize_users(data), f, indent=4)


def normalize_users(users):
    unique_users = []
    seen_usernames = set()

    for user in users:
        username = user.get("username", "").strip()
        if not username:
            continue

        username_key = username.lower()
        if username_key in seen_usernames:
            continue

        seen_usernames.add(username_key)
        unique_users.append({
            "username": username,
            "password": user.get("password", "")
        })

    return unique_users

def load_plans():
    try:
        with open("data/plans.json") as f:
            return json.load(f)
    except:
        return []

def save_plans(data):
    with open("data/plans.json", "w") as f:
        json.dump(data, f, indent=4)


def load_json_file(path_obj):
    try:
        with open(path_obj) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def update_current_plan_progress():
    plan_date = session.get("current_plan_date")
    user = session.get("user")

    if not user:
        return

    completed_sessions = session.get("completed_study_sessions", 0)
    total_sessions = session.get("total_study_sessions", 0)
    percent = 0 if total_sessions == 0 else round((completed_sessions / total_sessions) * 100)

    plans = load_plans()
    target_plan = None

    if plan_date:
        for plan in reversed(plans):
            if plan.get("user") == user and plan.get("date") == plan_date:
                target_plan = plan
                break

    if target_plan is None:
        for plan in reversed(plans):
            if plan.get("user") == user:
                target_plan = plan
                session["current_plan_date"] = plan.get("date")
                break

    if target_plan is None:
        return

    target_plan["completed_study_sessions"] = completed_sessions
    target_plan["total_study_sessions"] = total_sessions
    target_plan["progress_percent"] = percent
    save_plans(plans)


def parse_iso_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def parse_plan_date(value):
    if not value:
        return None

    parsed = parse_iso_datetime(value)
    if parsed:
        return parsed.date()

    return None


def build_reflection(plan):
    study_sessions = int(plan.get("study_sessions", 0))
    completed = int(plan.get("completed_study_sessions", 0))
    percent = int(plan.get("progress_percent", 0))

    if study_sessions == 0:
        return "No study blocks were scheduled that day. Create a short plan to build momentum."

    if percent >= 100:
        return "Excellent finish. You completed every planned study block."

    remaining = max(0, study_sessions - completed)
    if percent >= 70:
        return f"Strong effort. {remaining} block{'s' if remaining != 1 else ''} remained for full completion."

    return f"You started the day. {remaining} block{'s' if remaining != 1 else ''} are still available to recover tomorrow."


def convert_time(t):
    h, m = map(int, t.split(":"))
    return h + m / 60


# ---------- AUTH ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"].strip()

        if any(u["username"].lower() == username.lower() for u in users):
            return render_template("signup.html", error="That username already exists.")

        users.append({
            "username": username,
            "password": request.form["password"]
        })
        save_users(users)
        return redirect("/login")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        for u in users:
            if u["username"] == request.form["username"] and u["password"] == request.form["password"]:
                session["user"] = u["username"]
                return redirect("/")
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------- INPUT ----------
@app.route("/", methods=["GET", "POST"])
def input_page():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        count = max(1, int(request.form.get("numSubjects", 1)))
        selected_tier = request.form.get("pomodoroTier", "standard")
        study_minutes, break_minutes = POMODORO_TIERS.get(selected_tier, POMODORO_TIERS["standard"])

        subjects = []
        for i in range(1, count + 1):
            subjects.append({
                "name": request.form.get(f"sub{i}", ""),
                "difficulty": request.form.get(f"diff{i}", "easy"),
                "topics": int(request.form.get(f"topics{i}", 1))
            })

        time_slots = []
        for key in request.form:
            if key.startswith("start"):
                index = key.replace("start", "")
                start = convert_time(request.form[f"start{index}"])
                end = convert_time(request.form[f"end{index}"])
                if end < start:
                    end += 24
                time_slots.append((start, end))

        time_slots.sort(key=lambda slot: slot[0])

        schedule = generate_schedule(
            subjects,
            time_slots,
            study_minutes=study_minutes,
            break_minutes=break_minutes
        )
        schedule.sort(key=lambda item: item["start"])

        for item in schedule:
            item["duration_minutes"] = round((item["end"] - item["start"]) * 60)

        session["pomodoro"] = {
            "tier": selected_tier,
            "study_minutes": study_minutes,
            "break_minutes": break_minutes
        }

        for s in schedule:
            s["start"] = format_time(s["start"])
            s["end"] = format_time(s["end"])

        session["raw_schedule"] = copy.deepcopy(schedule)
        session["schedule"] = copy.deepcopy(schedule)
        session["total_study_sessions"] = sum(1 for item in schedule if item.get("type", "study") != "break")
        session["completed_study_sessions"] = 0
        session["completed_history"] = []

        # SAVE PLAN
        plans = load_plans()
        plan_date = str(datetime.now())
        plans.append({
            "user": session["user"],
            "date": plan_date,
            "pomodoro": session.get("pomodoro", {}),
            "schedule": schedule,
            "completed_study_sessions": 0,
            "total_study_sessions": session.get("total_study_sessions", 0),
            "progress_percent": 0,
            "productive_minutes": 0,
        })
        save_plans(plans)
        session["current_plan_date"] = plan_date

        return redirect("/schedule")

    return render_template("input.html")


# ---------- PAGES ----------
@app.route("/schedule")
def schedule():
    pomodoro = session.get("pomodoro", {"tier": "standard", "study_minutes": 50, "break_minutes": 10})
    return render_template("schedule.html", schedule=session.get("raw_schedule", session.get("schedule", [])), pomodoro=pomodoro)

@app.route("/progress")
def progress():
    if "user" not in session:
        return redirect("/login")

    study_only = [
        item for item in session.get("schedule", [])
        if item.get("type", "study") != "break"
    ]
    pomodoro = session.get("pomodoro", {"tier": "standard", "study_minutes": 50, "break_minutes": 10})
    next_session = study_only[0] if study_only else None
    current_study_minutes = (
        next_session.get("duration_minutes", pomodoro["study_minutes"]) if next_session else pomodoro["study_minutes"]
    )
    total_sessions = session.get("total_study_sessions", len(study_only))
    completed_sessions = session.get("completed_study_sessions", 0)
    percent = 0 if total_sessions == 0 else round((completed_sessions / total_sessions) * 100)

    return render_template(
        "progress.html",
        schedule=study_only,
        pomodoro=pomodoro,
        next_session=next_session,
        current_study_minutes=current_study_minutes,
        completed_sessions=completed_sessions,
        total_sessions=total_sessions,
        percent=percent,
    )


@app.route("/complete-session", methods=["POST"])
def complete_session():
    current_schedule = session.get("schedule", [])
    payload = request.get_json(silent=True) or {}
    target_subject = payload.get("subject")
    target_start = payload.get("start")
    target_end = payload.get("end")
    completed_item = None

    for index, item in enumerate(current_schedule):
        if item.get("type", "study") != "break":
            if target_subject and target_start and target_end:
                if (
                    item.get("subject") != target_subject or
                    item.get("start") != target_start or
                    item.get("end") != target_end
                ):
                    continue

            current_schedule.pop(index)
            completed_item = item
            session["completed_study_sessions"] = session.get("completed_study_sessions", 0) + 1
            completed_history = session.get("completed_history", [])
            completed_history.append({"item": item, "index": index})
            session["completed_history"] = completed_history
            break

    session["schedule"] = current_schedule
    update_current_plan_progress()

    plans = load_plans()
    for plan in reversed(plans):
        if plan.get("user") == session.get("user") and plan.get("date") == session.get("current_plan_date"):
            if completed_item:
                plan["productive_minutes"] = int(plan.get("productive_minutes", 0)) + int(completed_item.get("duration_minutes", session.get("pomodoro", {}).get("study_minutes", 0)))
            break
    save_plans(plans)

    study_only = [item for item in current_schedule if item.get("type", "study") != "break"]
    total_sessions = session.get("total_study_sessions", len(study_only))
    completed_sessions = session.get("completed_study_sessions", 0)
    percent = 0 if total_sessions == 0 else round((completed_sessions / total_sessions) * 100)
    next_session = study_only[0] if study_only else None

    if next_session and "duration_minutes" not in next_session:
        next_session["duration_minutes"] = session.get("pomodoro", {}).get("study_minutes", 50)

    current_pomodoro = session.get("pomodoro", {})
    current_study_minutes = next_session.get("duration_minutes", current_pomodoro.get("study_minutes", 50)) if next_session else current_pomodoro.get("study_minutes", 50)

    return jsonify({
        "completed_sessions": completed_sessions,
        "total_sessions": total_sessions,
        "percent": percent,
        "next_session": next_session,
        "current_study_minutes": current_study_minutes,
        "can_undo": bool(session.get("completed_history")),
        "remaining_subjects": [item.get("subject", "") for item in study_only],
    })


@app.route("/undo-session", methods=["POST"])
def undo_session():
    current_schedule = session.get("schedule", [])
    completed_history = session.get("completed_history", [])

    if not completed_history:
        total_sessions = session.get("total_study_sessions", len(current_schedule))
        completed_sessions = session.get("completed_study_sessions", 0)
        percent = 0 if total_sessions == 0 else round((completed_sessions / total_sessions) * 100)
        next_session = next((item for item in current_schedule if item.get("type", "study") != "break"), None)
        current_pomodoro = session.get("pomodoro", {})
        current_study_minutes = next_session.get("duration_minutes", current_pomodoro.get("study_minutes", 50)) if next_session else current_pomodoro.get("study_minutes", 50)

        return jsonify({
            "completed_sessions": completed_sessions,
            "total_sessions": total_sessions,
            "percent": percent,
            "next_session": next_session,
            "current_study_minutes": current_study_minutes,
            "can_undo": False,
            "restored_item": None,
            "remaining_subjects": [item.get("subject", "") for item in current_schedule if item.get("type", "study") != "break"],
        })

    last_completed = completed_history.pop()
    restored_item = last_completed.get("item")
    restore_index = last_completed.get("index", 0)

    if restored_item:
        restore_index = max(0, min(restore_index, len(current_schedule)))
        current_schedule.insert(restore_index, restored_item)
        session["completed_study_sessions"] = max(0, session.get("completed_study_sessions", 0) - 1)

    session["schedule"] = current_schedule
    session["completed_history"] = completed_history
    update_current_plan_progress()

    plans = load_plans()
    removed_minutes = 0
    if restored_item:
        removed_minutes = int(restored_item.get("duration_minutes", session.get("pomodoro", {}).get("study_minutes", 0)))

    for plan in reversed(plans):
        if plan.get("user") == session.get("user") and plan.get("date") == session.get("current_plan_date"):
            if restored_item:
                plan["productive_minutes"] = max(0, int(plan.get("productive_minutes", 0)) - removed_minutes)
            break
    save_plans(plans)

    study_only = [item for item in current_schedule if item.get("type", "study") != "break"]
    total_sessions = session.get("total_study_sessions", len(study_only))
    completed_sessions = session.get("completed_study_sessions", 0)
    percent = 0 if total_sessions == 0 else round((completed_sessions / total_sessions) * 100)
    next_session = study_only[0] if study_only else None

    if next_session and "duration_minutes" not in next_session:
        next_session["duration_minutes"] = session.get("pomodoro", {}).get("study_minutes", 50)

    current_pomodoro = session.get("pomodoro", {})
    current_study_minutes = next_session.get("duration_minutes", current_pomodoro.get("study_minutes", 50)) if next_session else current_pomodoro.get("study_minutes", 50)

    return jsonify({
        "completed_sessions": completed_sessions,
        "total_sessions": total_sessions,
        "percent": percent,
        "next_session": next_session,
        "current_study_minutes": current_study_minutes,
        "can_undo": bool(completed_history),
        "restored_item": restored_item,
        "remaining_subjects": [item.get("subject", "") for item in study_only],
    })

@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    plans = load_plans()
    user_plans = [p for p in plans if p["user"] == session["user"]]
    user_plans.sort(key=lambda p: parse_plan_date(p.get("date")) or date.min, reverse=True)

    history_plans = []
    for plan in user_plans:
        pomodoro = plan.get("pomodoro", {})
        study_minutes = 0
        break_minutes = 0
        study_sessions = 0
        break_sessions = 0

        for item in plan.get("schedule", []):
            item_type = item.get("type", "study")
            duration_minutes = item.get("duration_minutes")

            if duration_minutes is None:
                if item_type == "break" or item.get("subject") == "Break":
                    duration_minutes = pomodoro.get("break_minutes", 0)
                else:
                    duration_minutes = pomodoro.get("study_minutes", 0)

            duration_minutes = int(duration_minutes)

            if item_type == "break" or item.get("subject") == "Break":
                break_minutes += duration_minutes
                break_sessions += 1
            else:
                study_minutes += duration_minutes
                study_sessions += 1

        normalized_plan = {
            **plan,
            "study_minutes": study_minutes,
            "break_minutes": break_minutes,
            "study_sessions": study_sessions,
            "break_sessions": break_sessions,
            "completed_study_sessions": min(max(int(plan.get("completed_study_sessions", 0)), 0), study_sessions),
            "total_study_sessions": int(plan.get("total_study_sessions", study_sessions)),
            "progress_percent": int(
                plan.get(
                    "progress_percent",
                    0 if study_sessions == 0 else round((min(max(int(plan.get("completed_study_sessions", 0)), 0), study_sessions) / study_sessions) * 100)
                )
            ),
        }
        normalized_plan["reflection"] = build_reflection(normalized_plan)
        history_plans.append(normalized_plan)

    return render_template(
        "history.html",
        plans=history_plans,
    )


@app.route("/dashboard")
def ml_dashboard():
    if "user" not in session:
        return redirect("/login")

    root_dir = Path(__file__).resolve().parent
    reports_dir = root_dir / "artifacts" / "reports"
    plots_dir = root_dir / "artifacts" / "plots"
    models_dir = root_dir / "artifacts" / "models"

    step_descriptions = {
        1: "Define business objective for predicting study-session completion.",
        2: "Declare ML task type as binary classification.",
        3: "Set target label to session_completed.",
        4: "Define evaluation metrics: accuracy, precision, recall, and F1.",
        5: "Declare synthetic-only data source strategy.",
        6: "Generate and ingest synthetic study-session dataset.",
        7: "Run exploratory data analysis and export plots.",
        8: "Validate schema, nulls, duplicates, and value ranges.",
        9: "Check label distribution and target quality.",
        10: "Perform stratified train/validation/test split.",
        11: "Clean data and normalize column structures.",
        12: "Handle missing values using train-fitted imputations.",
        13: "Cap outliers with train-fitted IQR bounds.",
        14: "Create engineered features from study behavior signals.",
        15: "Apply preprocessing: scaling + encoding and save transformer.",
        16: "Train baseline logistic regression model.",
        17: "Train main random forest model.",
        18: "Tune hyperparameters using cross-validated grid search.",
        19: "Evaluate on test data with confusion matrix and feature importance.",
        20: "Package deployment bundle and simulate prediction API.",
    }

    step_rows = []
    completed_steps = 0
    for i in range(1, 21):
        matches = sorted(reports_dir.glob(f"step_{i:02d}*.json"))
        is_complete = bool(matches)
        if is_complete:
            completed_steps += 1

        step_rows.append(
            {
                "step": f"Step {i}",
                "status": "Completed" if is_complete else "Pending",
                "report": matches[0].name if is_complete else "-",
                "description": step_descriptions.get(i, "No description available."),
            }
        )

    required_artifacts = [
        reports_dir / "step_20_deployment.json",
        reports_dir / "step_19_evaluation.json",
        reports_dir / "step_20_sample_predictions.csv",
        models_dir / "deployment_bundle.joblib",
        models_dir / "tuned_random_forest.joblib",
        plots_dir / "evaluation_confusion_matrix.png",
    ]
    missing_artifacts = [str(path.relative_to(root_dir)) for path in required_artifacts if not path.exists()]
    pipeline_done = completed_steps == 20 and not missing_artifacts

    baseline_report = load_json_file(reports_dir / "step_16_baseline.json")
    main_report = load_json_file(reports_dir / "step_17_training.json")
    tuned_report = load_json_file(reports_dir / "step_18_tuning.json")
    eval_report = load_json_file(reports_dir / "step_19_evaluation.json")
    deploy_report = load_json_file(reports_dir / "step_20_deployment.json")

    metrics = {
        "baseline": baseline_report.get("metrics_validation", {}),
        "main": main_report.get("metrics_validation", {}),
        "tuned": tuned_report.get("metrics_validation", {}),
        "test": eval_report.get("metrics_test", {}),
    }

    confusion = eval_report.get("confusion_matrix", {})
    confusion_labels = ["TN", "FP", "FN", "TP"]
    confusion_values = [
        confusion.get("tn", 0),
        confusion.get("fp", 0),
        confusion.get("fn", 0),
        confusion.get("tp", 0),
    ]

    feature_rows = []
    feature_path = reports_dir / "step_19_feature_importance.csv"
    if feature_path.exists():
        with open(feature_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for idx, row in enumerate(reader):
                if idx >= 10:
                    break
                feature_rows.append(
                    {
                        "feature": row.get("feature", ""),
                        "importance": round(float(row.get("importance", 0)), 4),
                    }
                )

    sample_rows = []
    sample_path = reports_dir / "step_20_sample_predictions.csv"
    if sample_path.exists():
        with open(sample_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for idx, row in enumerate(reader):
                if idx >= 5:
                    break
                sample_rows.append(row)

    return render_template(
        "dashboard.html",
        step_rows=step_rows,
        completed_steps=completed_steps,
        pipeline_done=pipeline_done,
        missing_artifacts=missing_artifacts,
        metrics=metrics,
        confusion_labels=confusion_labels,
        confusion_values=confusion_values,
        feature_rows=feature_rows,
        sample_rows=sample_rows,
        deployment_status=deploy_report.get("api_simulation_status", "not_run"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)