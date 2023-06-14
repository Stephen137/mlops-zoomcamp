FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip 
RUN pip install pipenv 
RUN pip install pandas numpy scikit-learn
RUN pip install pyarrow

WORKDIR /app

COPY starter.py starter.py

COPY [ "model.bin", "model.bin"]

ENTRYPOINT [ "python", "starter.py" ]