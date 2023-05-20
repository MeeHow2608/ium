FROM ubuntu:latest

RUN apt update && apt install -y figlet
RUN apt-get update
RUN apt install python3-pip -y
RUN pip install --user pandas
RUN pip install --user jupyter
RUN pip install --user kaggle
RUN pip install --user seaborn
RUN pip install --user tensorflow
RUN pip install --user sacred
RUN pip install --user numpy
RUN pip install --user scikit-learn
RUN apt install git -y
COPY train.py /app/train.py
CMD python3 predict.py
