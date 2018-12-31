opt_rcse:
	g++ rcse.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse
debug_rcse:
	g++ -g rcse.cpp -fopenmp -Wall -std=c++11 -o debug_rcse
clext_opt_rcse:
	g++ rcse_clext.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse_clext
clext_debug_rcse:
	g++ -g rcse_clext.cpp -fopenmp -Wall -std=c++11 -o debug_rcse_clext

clean:
	rm -f *.o opt_rcse
	rm -f *.o debug_rcse
	rm -f *.o opt_rcse_clext
	rm -f *.o debug_rcse_clext
