FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx

RUN python -m pip install --upgrade pip
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt
RUN pip install opencv-python==4.4.0.44
RUN pip install Pillow==8.0.1

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python"]

COPY examples/joint_inference/online_course  /home/work

CMD ["big_model.py"]
