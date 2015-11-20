#!/bin/bash
#To run the bash script, use: $source reports_gen.sh
# Create directory to save the reports

mkdir -p current_reports

# To generate the general hotspots report

amplxe-cl -report hotspots -r rhs/ > ~/path/current_reports/hotspots.txt

### To generate the concurrency report

amplxe-cl -report callstacks -r rcc/ -group-by callstack > ~/path/current_reports/cc_callstack.txt

amplxe-cl -report callstacks -r rcc/ -group-by function > ~/path/current_reports/cc_function.txt

amplxe-cl -report callstacks -r rcc/ -group-by function-callstack > ~/path/current_reports/cc_fc.txt

rm -r rhs
rm -r rcc

