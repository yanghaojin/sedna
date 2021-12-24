FROM python:3.6-slim

RUN apt update \
  && apt install -y libgl1-mesa-glx git

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

RUN pip install -r /home/requirements.txt
RUN pip install torch torchvision torchaudio

ENV PYTHONPATH "/home/lib"

COPY ./lib /home/lib
#RUN git clone https://github.com/TL-System/plato.git /home/plato
#RUN cd /home/plato && git reset --hard dac991791ac82be5129a7e16a0695a55e40aa4ca
#RUN rm -rf /home/plato/.git
#RUN pip install -r /home/plato/requirements.txt
#RUN  pip install plato-learn==0.2.7

WORKDIR /home/work
COPY examples/federated_learning/online_course/pytorch  /home/work/

ENTRYPOINT ["python", "train_on_kaggle.py"]
