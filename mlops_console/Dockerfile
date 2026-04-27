FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops_console.py .

EXPOSE 8503

CMD ["streamlit", "run", "mlops_console.py", "--server.port=8503", "--server.address=0.0.0.0"]