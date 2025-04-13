# upload_server/main.py
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from .fl_logic import start_fl_server  # Fixed: Added relative import with dot
# Assuming authentication setup (e.g., verifying Supabase JWT)
# from .auth import get_current_active_user, User  # Placeholder for auth

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase URL and Service Key must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

class DatasetInfo(BaseModel):
    name: str
    description: str | None = None
    data_type: str # e.g., 'tabular_csv'
    # Add other fields matching 'datasets' table if needed

@app.post("/upload_dataset/")
async def upload_dataset(
    info: DatasetInfo = Depends(),
    file: UploadFile = File(...),
    # current_user: User = Depends(get_current_active_user) # Enable auth
):
    # Authentication/Authorization check: Ensure user has 'data_provider' role
    # user_profile = supabase.table("user_profiles").select("role").eq("id", current_user.id).execute()
    # if not user_profile.data or user_profile.data[0]['role'] != 'data_provider':
    #     raise HTTPException(status_code=403, detail="User not authorized to upload data")

    # --- Option 1: Store file in Supabase Storage ---
    # file_path = f"datasets/{current_user.id}/{info.name}/{file.filename}"
    # try:
    #     # Note: supabase-py might not directly support streaming uploads well.
    #     # Consider using Supabase JS client via edge function or direct HTTP requests
    #     # For simplicity here, we assume direct upload (may load into memory)
    #     content = await file.read()
    #     supabase.storage.from_("datasets").upload(file_path, content)
    #     # Get public URL or signed URL if needed
    #     # storage_url = supabase.storage.from_("datasets").get_public_url(file_path)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to upload to storage: {e}")

    # --- Option 2: Store metadata only (assuming data is elsewhere or uploaded manually) ---
    # storage_url = "manual_upload_or_external_link" # Or None

    # Insert metadata into the datasets table
    try:
        insert_data = {
            "name": info.name,
            "description": info.description,
            # "data_provider_id": current_user.id,
            "data_provider_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479", # Hardcoded placeholder UUID
            "size_bytes": file.size, # Approximate size
            "data_type": info.data_type,
            "status": "pending", # Requires admin approval maybe?
            # "metadata": {"storage_path": file_path, "original_filename": file.filename} # If using storage
            "metadata": {"original_filename": file.filename}
        }
        result = supabase.table("datasets").insert(insert_data).execute()

        if not result.data:
             raise HTTPException(status_code=500, detail=f"Failed to insert dataset metadata: {getattr(result, 'error', 'Unknown error')}")

        return {"message": "Dataset uploaded successfully", "dataset_id": result.data[0]['id']} # , "storage_path": file_path}

    except Exception as e:
        # Consider cleanup if storage upload succeeded but DB insert failed
        raise HTTPException(status_code=500, detail=f"Failed to insert dataset metadata: {e}")

# Add routes for listing datasets, approving, etc.



@app.post("/start_fl")
async def trigger_fl_training(rounds: int = 3):
    # Could run start_fl_server in a background task
    print(f"Received request to start FL training for {rounds} rounds.")
    # In a real scenario, use background tasks (e.g., Celery, asyncio.create_task)
    # For simplicity, running directly here (blocks the request)
    start_fl_server(num_rounds=rounds)
    return {"message": "Federated Learning process initiated."}
# Add other API endpoints as needed (e.g., get status, list models)