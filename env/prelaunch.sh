#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if grep -q "^nv_ring" /proc/modules; then
    sudo rmmod nv_ring
fi

sudo insmod $SCRIPT_DIR/driver/nv_ring.ko
sudo chown orinagx /dev/nv_sensor_ring*
echo "PCIe Setup complete."

sudo busybox devmem 0x0c303018 w 0xc458 # CAN0 DIN
sudo busybox devmem 0x0c303010 w 0xc400 # CAN0 DOUT

sudo busybox devmem 0x0c303008 w 0xc458 # CAN1 DIN
sudo busybox devmem 0x0c303000 w 0xc400 # CAN1 DOUT

sudo ip link set can0 down
sudo ip link set can1 down

sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 berr-reporting on fd on
sudo ip link set can1 up type can bitrate 500000 dbitrate 1000000 berr-reporting on fd on

# sudo cat /proc/device-tree/bus@0/mttcan@c310000/status
# sudo cat /proc/device-tree/bus@0/mttcan@c320000/status

echo "CAN FD Setup complete."