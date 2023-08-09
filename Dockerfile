FROM python:3.8.10
EXPOSE 8501
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "src/app.py"]