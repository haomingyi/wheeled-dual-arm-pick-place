#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Snap-launched terminals can inject core20 GTK / locale libraries that conflict
# with system Gazebo. Clear them before launching Gazebo.
for name in ${!SNAP@}; do
  unset "$name"
done
unset GTK_PATH
unset LOCPATH

GZ_VENDOR_LIBS="$(find /opt/ros/jazzy/opt -maxdepth 2 -type d -path '*/lib' | paste -sd: -)"
export GZ_CONFIG_PATH="/opt/ros/jazzy/opt/gz_sim_vendor/share/gz:/opt/ros/jazzy/opt/sdformat_vendor/share/gz"
export LD_LIBRARY_PATH="${GZ_VENDOR_LIBS}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu"

WORLD="sim_gazebo/worlds/mobile_base_minimal.sdf"

if [[ "${1:-}" == "--server-only" ]]; then
  exec gz sim -s -r -v 4 "$WORLD"
fi

exec gz sim "$WORLD"
