FROM python:3.7.5-buster

ADD ./requirements.txt /
RUN --mount=type=cache,id=custom-aptget,target=.cache/aptget apt-get update && apt-get install -y python3-opencv
RUN --mount=type=cache,id=custom-pip,target=.cache/pip pip3 install -r requirements.txt

COPY ./aihandler /aihandler
COPY ./backend /backend
COPY ./frontend /frontend
COPY ./scripts /scripts
RUN mkdir -p /aiworkingdir
EXPOSE 8080
EXPOSE 8004

WORKDIR /scripts
RUN chmod +x run_all_modules.sh
RUN chmod +x serve-backend.sh
RUN chmod +x serve-front.sh
ENTRYPOINT ["./run_all_modules.sh"]
