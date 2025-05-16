#!/usr/bin/env bash

###############################################################################
# profiler.sh:
#  A low-overhead memory profiler that spawns a command, periodically samples
#  its RSS (resident set size) from /proc, and logs timestamp + RSS to a file.
#
# Usage:
#
#   profiler.sh [options] -- <command> [args...]
#
# Options (all optional):
#   -r|--rate <seconds>         Sampling rate (sleep time). Default: 0.0001
#   -i|--stdin-file <path>      File to feed as stdin to the command. If not
#                               provided, no stdin redirection is used.
#   -o|--output-file <path>     Output file for the profiler data. Default:
#                               /tmp/profiler.log
#   -m|--memory-limit <mb>      Memory limit in MB. If not provided or set to 0, no memory limit is enforced.
#                               /tmp/stdin  (per your request; feel free
#                               to rename to /tmp/profiler.log)
#
# Example:
#   profiler.sh -r 0.001 -i input.txt -o /tmp/my_profiler.log -- myprog --flag1
###############################################################################

###############################################################################
# Default settings
###############################################################################
SAMPLING_RATE="0.0001"
STDIN_FILE=""
OUTPUT_FILE="/tmp/profiler.log"  # Adjust the default path as desired
MEMORY_LIMIT_MB=0  # Default: no memory limit (0 means no limit)

###############################################################################
# Parse arguments until we see '--'
###############################################################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--rate)
            SAMPLING_RATE="$2"
            shift 2
            ;;
        -i|--stdin-file)
            STDIN_FILE="$2"
            shift 2
            ;;
        -o|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -m|--memory-limit)
            MEMORY_LIMIT_MB="$2"
            shift 2
            ;;
        --)
            # The double-dash signals the end of our options
            shift
            break
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [options] -- <command> [args...]"
            exit 1
            ;;
    esac
done

###############################################################################
# Check if we have at least one argument after '--' (the command)
###############################################################################
if [ $# -lt 1 ]; then
  echo "Error: no command specified."
  echo "Usage: $0 [options] -- <command> [args...]"
  exit 1
fi

###############################################################################
# Prepare the output file
###############################################################################
rm -f "$OUTPUT_FILE"

###############################################################################
# Cache the page size (in kilobytes) outside of the sampling loop
###############################################################################
PAGE_SIZE_BYTES=$(getconf PAGESIZE)
PAGE_SIZE_KB=$(( PAGE_SIZE_BYTES / 1024 ))

###############################################################################
# Start the command in the background (optionally redirect stdin if requested)
###############################################################################
if [ -n "$STDIN_FILE" ]; then
  "$@" < "$STDIN_FILE" &
else
  "$@" &
fi
PID=$!

###############################################################################
# Ensure the target PID is alive before we start sampling
###############################################################################
while ! kill -0 "$PID" 2>/dev/null; do
    sleep 0.0001
done

###############################################################################
# Determine if we need to sleep (pre-calculate once outside the loop)
SHOULD_SLEEP=1
if (( $(echo "$SAMPLING_RATE == 0" | bc -l) )); then
    SHOULD_SLEEP=0
fi

# Convert memory limit from MB to KB for comparison
ENFORCE_MEMORY_LIMIT=0
[ "$MEMORY_LIMIT_MB" -gt 0 ] && MEMORY_LIMIT_KB=$(( MEMORY_LIMIT_MB * 1024 )) && ENFORCE_MEMORY_LIMIT=1

###############################################################################
# Sampling loop until the process exits
###############################################################################
results=()

EXIT_CODE=0

while kill -0 "$PID" 2>/dev/null; do
    # Timestamp in nanoseconds
    timestamp_ns=$(date +%s%N)
    
    # Read the resident set size from /proc/<PID>/statm
    if [ -r /proc/"$PID"/statm ]; then
        # shellcheck disable=SC2034
        read -r size resident share text lib data dt < /proc/"$PID"/statm 2>/dev/null
        # Convert resident pages to kilobytes
        rss_kb=$(( resident * PAGE_SIZE_KB ))
    else
        # If the file is no longer readable, the process likely exited
        rss_kb=0
    fi
    
    # Check if memory limit exceeded
    if [ "$ENFORCE_MEMORY_LIMIT" -eq 1 ] && [ "$rss_kb" -gt "$MEMORY_LIMIT_KB" ]; then
        echo "Process exceeded memory limit ($MEMORY_LIMIT_MB MB), killing..." >&2
        # Kill the process
        kill -9 "$PID" 2>/dev/null
        # Set exit code for OOM
        EXIT_CODE=137
        break
    fi
    
    # Add to our in-memory array
    results+=("$timestamp_ns $rss_kb")
    
    # Sleep by the specified sampling rate only if not zero
    if [ $SHOULD_SLEEP -eq 1 ]; then
        sleep "$SAMPLING_RATE"
    fi
done

###############################################################################
# Wait for the profiled program to fully exit
###############################################################################
wait "$PID" 2>/dev/null

###############################################################################
# Write results to disk
###############################################################################
printf "%s\n" "${results[@]}" > "$OUTPUT_FILE"

# Exit with appropriate code
exit $EXIT_CODE
