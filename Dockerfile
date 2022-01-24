FROM python:3.7.5-buster

COPY ./aihandler /aihandler
COPY ./backend /backend
COPY ./frontend /frontend
COPY ./scripts /scripts
ADD ./requirements.txt /
RUN mkdir -p /aiworkingdir
RUN --mount=type=cache,id=custom-aptget,target=/root/.cache/aptget apt-get update && apt-get install -y python3-opencv
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install -r requirements.txt
EXPOSE 8080
EXPOSE 8004

WORKDIR /scripts
ENTRYPOINT ["./run_all_modules.sh"]
