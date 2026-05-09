# Vitis HLS build script for the rectify / remap kernel.
# Usage from hls/:
#   vitis_hls -f run_rectify_hls.tcl csim
#   vitis_hls -f run_rectify_hls.tcl csynth
#   vitis_hls -f run_rectify_hls.tcl cosim
#   vitis_hls -f run_rectify_hls.tcl

set PROJECT "rectify_hls"
set SOLUTION "sol1"
set PART "xczu7ev-ffvc1156-2-e"
set PERIOD 5

set action "all"
if {$argc > 0} {
    set action [lindex $argv 0]
}

open_project -reset $PROJECT
set_top rectify_kernel
add_files rectify.cpp -cflags "-I."
add_files rectify.h -cflags "-I."
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
    all - default {
        csim_design
        csynth_design
        cosim_design -rtl verilog
    }
}

exit
