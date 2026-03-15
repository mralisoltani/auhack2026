pip install -r requirements.txt
streamlit run dashboard/app.py

Docker Usage
Local mode (default):

Run the dashboard as usual; training and inference run in-process.
API mode:

Start the ML API: docker compose up ml-api -d
In the dashboard, choose API (ML API) in the Prediction tab
Train and predict via the ML API
Full stack:

docker compose up
ML API: http://localhost:8000
Dashboard: http://localhost:8501


Using Docker Compose (recommended)
Build and start all services:

docker compose up --build
Builds images if needed
Starts ml-api (port 8000) and dashboard (port 8501)
Streams logs in the foreground
Run in the background:

docker compose up --build -d
Same as above, but detached (background)
Use docker compose logs -f to follow logs
Start only the ML API:

docker compose up ml-api -d
Runs only the ML API
Dashboard can run locally with streamlit run dashboard/app.py and use the API
Stop services:

docker compose down
Using plain Docker
ML API only (from project root):

docker build -f ml_service/Dockerfile -t ml-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models -e MODEL_DIR=/app/models ml-api
Dashboard only:

docker build -f Dockerfile.dashboard -t dashboard .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -e ML_API_URL=http://host.docker.internal:8000 dashboard
(Use host.docker.internal so the container can reach the ML API on the host.)