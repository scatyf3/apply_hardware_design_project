# Vitis HLS build script for the resize kernel.
#
# Usage (from this directory):
#   vitis_hls -f run_hls.tcl          # C-sim + csynth + co-sim
#   vitis_hls -f run_hls.tcl csim     # only C simulation
#   vitis_hls -f run_hls.tcl csynth   # only C synthesis
#   vitis_hls -f run_hls.tcl cosim    # only co-simulation (needs csynth first)
#
# The target part/clock match the PYNQ-Z2 (Zynq-7020) board; adjust
# `PART` and `PERIOD` for your actual target.

set PROJECT "resize_hls"
set SOLUTION "sol1"
set PART     "xc7z020clg400-1"
set PERIOD   10

set action "all"
if {$argc > 0} {
    set action [lindex $argv 0]
}

open_project -reset $PROJECT
set_top resize_kernel
add_files resize.cpp -cflags "-I."
add_files resize.h   -cflags "-I."
add_files -tb resize_tb.cpp -cflags "-I."

open_solution -reset $SOLUTION
set_part $PART
create_clock -period $PERIOD -name default

switch -- $action {
    csim {
        csim_design
    }
    csynth {
        csynth_design
    }
    cosim {
        cosim_design -rtl verilog
    }
    all -
    default {
        csim_design
        csynth_design
        cosim_design -rtl verilog
    }
}

exit
