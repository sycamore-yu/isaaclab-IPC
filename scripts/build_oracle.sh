#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/oracle
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/native:/native:ro" -v "$PWD/results/oracle:/out" ipc-hku-mars:noetic \
  bash -c 'g++ -O2 -I/usr/include/eigen3 -I/usr/local/include -I/usr/local/include/osqp /native/qp_oracle.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lOsqpEigen -losqp -o /out/qp_oracle'
