FROM tensorflow/tensorflow:2.3.0

RUN apt update \
  && apt install -y libgl1-mesa-glx git

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

RUN pip install -r /home/requirements.txt
RUN pip install keras
RUN pip install tensorflow-datasets

ENV PYTHONPATH "/home/lib:/home/plato"

COPY ./lib /home/lib
RUN git clone https://github.com/TL-System/plato.git /home/plato
RUN cd /home/plato && git reset --hard dac991791ac82be5129a7e16a0695a55e40aa4ca
RUN rm -rf /home/plato/.git
RUN pip install -r /home/plato/requirements.txt
#RUN  pip install plato-learn==0.2.7

WORKDIR /home/work
COPY examples/federated_learning/online_course/tensorflow  /home/work/

CMD ["/bin/sh", "-c", "ulimit -n 50000; python aggregate.py"]
