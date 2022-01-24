#!/bin/bash

./serve-backend.sh > "backlog" & 

./serve-front.sh > "frontlog" &

wait -n
