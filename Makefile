# ------------------------------
# ClusterVision Makefile
# ------------------------------

run:
	streamlit run app.py

install:
	pip install -r requirements.txt

docker-build:
	docker build -t clustervision .

docker-run:
	docker run -p 8501:8501 clustervision
