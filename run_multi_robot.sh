#!/bin/bash

# this script is to automate running several instances of the SocNavBench simulator with different algorithms on the same test
# note, to change the test, edit the individual test in the episode_params_val.ini file (top of the file)


# make output logs directory
logdir=tests/socnav/run_multi_robot_logs
mkdir -p $logdir

# define useful utilities

py="PYTHONPATH='.' python3" # how to execute python
# don't print the extra characters to print colours on tty
ignore_colours='s/\x1B\[[0-9;]\{1,\}[A-Za-z]//g' # source: https://stackoverflow.com/a/51141872
# only print last line of the \r carriage return lines (make noop to see everything)
manage_overwrite='BEGIN { RS="[\r\n]" } {a = $0 substr(a, 1 + length)} RT ~ /\n/ {print a; a=""}' # source: https://unix.stackexchange.com/a/522262

find_pid() {
    unique_info=$1
    # there is probably a more efficient way to do this but it is late and I'm tired
    echo $(ps ax | grep -v grep | grep "${unique_info}" | awk '{print $1;}')
}

# auxiliary executables
RVO_exec="./joystick/RVO2/RVO2"
SF_exec="./joystick/social_force/social_force"

# can manually add/remove from this list
algos=("Sampling" "RVO" "RVOwCkpt" "social_forces" "Sacadrl" "SacadrlwCkpt")
for algo in ${algos[*]}; do

    # first off, start the socnavbench simulator 
    mkdir -p $logdir/$algo
    outdir=$logdir/$algo
    eval $py tests/test_episodes.py | \
        awk "$manage_overwrite" | \
        sed "$ignore_colours" \
        > $outdir/simulator.log & # log to file
    # sim_pid=$! # always 1 less than the correct value (probably bc of the redirect?)
    sim_pid=$(find_pid "tests/test_episodes.py")
    echo -e "Started simulator server for algo \"$algo\" (pid: $sim_pid)..."
    # wait a bit to start the joystick
    sleep 1.5

    export TF_CPP_MIN_LOG_LEVEL=3 # minimal TF logging
    eval $py joystick/joystick_client.py --algo "$algo" | \
        awk "$manage_overwrite" | \
        sed "$ignore_colours" \
        &> $outdir/joystick.log & # log to file (with stderr)
    joystick_pid=$(find_pid "joystick/joystick_client.py")
    echo -e "Started joystick client for algo \"$algo\" (pid: $joystick_pid)..."

    # auxiliary processes:
    sleep 1.5

    run_aux_exec () {
        executable=$1
        eval $executable \
            > $outdir/aux.log & # log auxiliary executable to file
        aux_pid=$(find_pid "${executable}")
        echo -e "Started \"$executable\" executable (pid: $aux_pid)..."
        tail --pid=$joystick_pid -f /dev/null
        echo -e "Joystick \"$algo\" completed"
        sleep 0.5
        kill -2 $aux_pid 2> /dev/null # gracefully interrupt (SIGINT) instead of hard kill (-9)
        kill $aux_pid 2> /dev/null # another kill for good measure
        echo -e "Stopped \"$executable\" executable"
        
    }

    if [[ $algo == "RVO" ]] || [[ $algo == "RVOwCkpt" ]]; then
        run_aux_exec $RVO_exec
    elif [[ $algo == "social_forces" ]]; then
        export LD_LIBRARY_PATH=./joystick/social_force/src 
        run_aux_exec $SF_exec
    fi

    # essentially "waits" for the simulator pid to finish
    tail --pid=$sim_pid -f /dev/null # only one simulator running at a time
    echo -e "Finished process for algo \"$algo\""
    printf "\n\n"
done
