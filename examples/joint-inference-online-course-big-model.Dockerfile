FROM python:3.6-slim

# using apt-get instead of apt for it being designed for terminal.
RUN apt-get update \
  && apt-get install -y libgl1-mesa-glx

RUN python -m pip install --upgrade pip
RUN pip install torch torchvision torchaudio

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt
ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib
COPY examples/joint_inference/online_course  /home/work

ENTRYPOINT ["python"]
CMD ["big_model.py"]
