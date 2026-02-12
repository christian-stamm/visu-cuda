#!/bin/bash
set -e

sudo modprobe can
sudo modprobe can_raw
sudo modprobe vcan
sudo modprobe can-gw

sudo ip link del vcan0 2>/dev/null || true
sudo ip link del vcan1 2>/dev/null || true

sudo ip link add dev vcan0 type vcan berr-reporting on fd on
sudo ip link add dev vcan1 type vcan berr-reporting on fd on
sudo ip link set up vcan0
sudo ip link set up vcan1

sudo cangw -F || true
sudo cangw -A -s vcan0 -d vcan1 -e
sudo cangw -A -s vcan1 -d vcan0 -e