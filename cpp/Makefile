opt_rcse:
	g++ rcse.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse
	#g++ rcse_117_avg.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse_avg

debug_rcse:
	g++ -g rcse.cpp -fopenmp -Wall -std=c++11 -o debug_rcse

opt_rcse_continual:
	g++ rcse_clext.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse_continual
    
debug_rcse_continual:
	g++ -g rcse_clext.cpp -fopenmp -Wall -std=c++11 -o debug_rcse_continual

opt_rcse_continual_multE:
	g++ rcse_clext_multE.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse_continual_multE

debug_rcse_continual_multE:
	g++ -g rcse_clext_multE.cpp -fopenmp -Wall -std=c++11 -o debug_rcse_continual_multE

opt_rcse_continual_multE_train:
	g++ rcse_clext_multE_train.cpp -fopenmp -Wall -O3 -std=c++11 -o opt_rcse_continual_multE_train

clean_logs:
	rm -f *.txt logfile.txt
	rm -f *.txt interactive_log.txt

clean:
	rm -f *.o opt_rcse
	rm -f *.o debug_rcse
	rm -f *.o opt_rcse_continual
	rm -f *.o debug_rcse_continual
	rm -f *.o opt_rcse_continual_multE
	rm -f *.o debug_rcse_continual_multE