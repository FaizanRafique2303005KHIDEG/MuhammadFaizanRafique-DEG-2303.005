FROM python:3.8-slim-buster

WORKDIR /home/app/

COPY ./ /home/app/

VOLUME /app/save_data

RUN pip install --no-cache-dir -r requirements.txt 

ENV PYTHONPATH=${PYTHONPATH}:/home/app/

CMD ["gunicorn", "main:app", "-b", "0.0.0.0:5000"]
