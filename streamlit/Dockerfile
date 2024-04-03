FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# run the Streamlit app
CMD ["streamlit", "run", "streamlit_code.py", "--server.port=8501", "--server.address=0.0.0.0"]
