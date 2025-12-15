#!/bin/bash
set -e

# If pyrt build doesn't exist or is empty (was overwritten by mount), symlink to backup
if [ ! -f "/app/pbrnerf/code/pyrt/build/pyrt.so" ]; then
    echo "Creating symlink to pyrt build from /opt/pyrt_build_backup..."
    # Remove empty build directory if it exists
    rm -rf /app/pbrnerf/code/pyrt/build
    # Create symlink to the backup (keeps build artifacts inside container only!)
    ln -sf /opt/pyrt_build_backup /app/pbrnerf/code/pyrt/build
fi

# Execute the command passed to the container
exec "$@"
