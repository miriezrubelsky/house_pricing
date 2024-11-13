# Use the official Python image from the Docker Hub
FROM python:3.9 

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip 
    #copy to code directory
COPY . /code    


RUN chmod +x /code/src

RUN pip install --no-cache-dir --upgrade -r code/src/house_pricing/requirements.txt

# Declare the directory as a volume
#VOLUME /code/src/churn_prediction_pipeline/trained_model
# Declare the directory as a volume
#VOLUME /code/src/churn_prediction_pipeline/logs



# Create the input-files directory
#RUN mkdir -p /input-files


# Dockerfile
#RUN mkdir -p /output-files && chmod -R 777 /output-files


EXPOSE 8000

WORKDIR /code

ENV PYTHONPATH "${PYTHONPATH}:/code/src"



ENTRYPOINT ["python", "src/house_pricing/main.py"]




#CMD ["pytest"]

# Default command to provide flexibility if no arguments are passed
CMD ["--help"]

#CMD ["--host", "0.0.0.0", "--port", "8000"]


