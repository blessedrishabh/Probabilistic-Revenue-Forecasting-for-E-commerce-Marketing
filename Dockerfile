From python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir-r requirements.txt

CMD ["streamlit", "run", "ui_backedn/streamlit_app/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]