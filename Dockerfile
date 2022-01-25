FROM python:3.7.5-buster

ADD ./requirements.txt /
RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 install -r requirements.txt

COPY ./aihandler /aihandler
COPY ./backend /backend
COPY ./frontend /frontend
COPY ./scripts /scripts
RUN mkdir -p /aiworkingdir
EXPOSE 8080
EXPOSE 8004

WORKDIR /scripts
ENTRYPOINT ["./run_all_modules.sh"]
