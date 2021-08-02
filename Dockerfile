FROM python:3.9-slim-buster

COPY . /app

WORKDIR /app

RUN apt update 

RUN apt -y install libgl1-mesa-glx

RUN apt -y install tesseract-ocr 

RUN apt -y install libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run" ]

CMD ["anpr_webapp.py"]


