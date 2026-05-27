.PHONY: help install data train test test-backend test-ml run-backend run-frontend docker-up docker-down clean

help:
	@echo "FirstWave commands:"
	@echo "  make install         Install all Python dependencies"
	@echo "  make data            Generate synthetic prescriber data"
	@echo "  make train           Train ML models (XGBoost + Cox PH + SHAP)"
	@echo "  make test            Run all tests (backend + ML pipeline)"
	@echo "  make test-backend    Run backend API tests only"
	@echo "  make test-ml         Run ML pipeline tests only"
	@echo "  make run-backend     Run FastAPI server on :8000"
	@echo "  make run-frontend    Run React dev server on :5173"
	@echo "  make docker-up       Run full stack via docker-compose"
	@echo "  make docker-down     Stop docker-compose stack"
	@echo "  make clean           Remove generated artifacts"

install:
	pip install -r backend/requirements.txt
	pip install -r ml/requirements.txt

data:
	python data/generate_synthetic_data.py

train:
	python ml/train.py

test: test-ml test-backend

test-backend:
	python -m pytest backend/tests/ -v

test-ml:
	python -m pytest tests/test_ml_pipeline.py -v

run-backend:
	cd $(CURDIR) && PYTHONPATH=$(CURDIR) uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	cd frontend && npm install && npm run dev

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	rm -rf data/processed/*.parquet data/processed/*.pkl
	rm -rf ml/models/*.pkl ml/models/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
