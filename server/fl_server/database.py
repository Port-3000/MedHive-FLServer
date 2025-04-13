# fl_server/database.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase URL and Service Key must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def get_supabase_client():
    return supabase

# Add helper functions to interact with Supabase tables
# e.g., create_fl_round, get_model_info, update_participant_status, etc.
# Example:
def create_fl_round_in_db(model_id: int, round_number: int):
    client = get_supabase_client()
    try:
        res = client.table("fl_rounds").insert({
            "model_id": model_id,
            "round_number": round_number,
            "status": "pending"
        }).execute()
        if res.data:
            return res.data[0]['id']
        else:
            print(f"Error creating FL round: {res.error}")
            return None
    except Exception as e:
        print(f"DB Exception creating FL round: {e}")
        return None

def update_round_status(round_id: int, status: str):
     client = get_supabase_client()
     try:
        client.table("fl_rounds").update({"status": status}).eq("id", round_id).execute()
     except Exception as e:
        print(f"DB Exception updating round status: {e}")

def update_aggregated_model_path(round_id: int, path: str):
    client = get_supabase_client()
    try:
        client.table("fl_rounds").update({"aggregated_model_path": path, "status": "completed", "end_time": "now()"}).eq("id", round_id).execute()
    except Exception as e:
        print(f"DB Exception updating agg model path: {e}")

def log_model_performance(model_id: int, round_id: int, metrics: dict, test_dataset_id: int | None = None):
    client = get_supabase_client()
    try:
        client.table("model_performance").insert({
            "model_id": model_id,
            "round_id": round_id,
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1_score"),
            "precision_score": metrics.get("precision"),
            "recall_score": metrics.get("recall"),
            "loss": metrics.get("loss"),
            "test_dataset_id": test_dataset_id
        }).execute()
    except Exception as e:
        print(f"DB Exception logging performance: {e}")

def get_active_model_info(model_name: str, model_version: str):
    client = get_supabase_client()
    try:
        res = client.table("ml_models").select("id, model_path").eq("name", model_name).eq("version", model_version).eq("status", "active").limit(1).execute()
        if res.data:
            return res.data[0]
        else:
            print(f"No active model found for {model_name} v{model_version}")
            return None
    except Exception as e:
        print(f"DB Exception getting active model: {e}")
        return None