# Vitis HLS build script for the rectify kernel.
#
# Usage (from this directory):
#   vitis_hls -f run_hls_rectify.tcl          # C-sim + csynth + co-sim
#   vitis_hls -f run_hls_rectify.tcl csim     # only C simulation
#   vitis_hls -f run_hls_rectify.tcl csynth   # only C synthesis
#   vitis_hls -f run_hls_rectify.tcl cosim    # only co-simulation (needs csynth first)
#
# Target part and clock match run_hls.tcl (Zynq UltraScale+); adjust for your
# actual board.

set PROJECT "rectify_hls"
set SOLUTION "sol1"
set PART     "xczu7ev-ffvc1156-2-e"
set PERIOD   5

set action "all"
if {$argc > 0} {
    set action [lindex $argv 0]
}

open_project -reset $PROJECT
set_top rectify_kernel
add_files rectify.cpp  -cflags "-I."
add_files rectify.h    -cflags "-I."
add_files resize.h     -cflags "-I."   ;# shared pixel_t / dim_t / MAX constants
add_files -tb rectify_tb.cpp -cflags "-I."

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
