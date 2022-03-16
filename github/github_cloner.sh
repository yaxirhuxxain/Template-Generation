#!/usr/bin/env bash
projects="listOfProjectsToClone"
while read project
do
  echo "Start to clone $project"
  git clone $project 
  sleep 1
done < $projects
