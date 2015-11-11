make realclean
rm *.optrpt 
rm Z_path*
make
qsub mine.pbs
