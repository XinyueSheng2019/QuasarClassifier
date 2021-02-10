int kali::sf(int numCadences, double dt, double *tIn, double *xIn, double *yIn, double *yerrIn, double *maskIn, double *lagVals, double *sfVals, double *sfErrVals) {
	#ifdef DEBUG_SF
		printf("numCadences: %d\n",numCadences);
		printf("dt: %f\n",dt);
	#endif
    #pragma omp parallel for default(none) shared(numCadences, sfVals, sfErrVals)
    for (int i = 0; i < numCadences; ++i) {
        sfVals[i] = 0.0;
        sfErrVals[i] = 0.0;
    }
    double count = 0.0;
    #pragma omp parallel for default(none) shared(numCadences, dt, tIn, yIn, yerrIn, maskIn, lagVals, sfVals, sfErrVals, count)
	for (int lagCad = 1; lagCad < numCadences; ++lagCad) {
		lagVals[lagCad] = lagCad*dt;
		count = 0.0;
		for (int pointNum = 0; pointNum < numCadences - lagCad; ++pointNum) {
			sfVals[lagCad] = sfVals[lagCad] + maskIn[pointNum]*maskIn[pointNum + lagCad]*pow((yIn[pointNum + lagCad] - yIn[pointNum]), 2.0);

			sfErrVals[lagCad] = sfErrVals[lagCad] + 2.0*maskIn[pointNum]*maskIn[pointNum + lagCad]*pow((yIn[pointNum + lagCad] - yIn[pointNum]), 2.0)*(pow(yIn[pointNum + lagCad], 2.0) + pow(yIn[pointNum], 2.0));
			
			count = count + maskIn[pointNum]*maskIn[pointNum + lagCad];
            #ifdef DEBUG_SF
				if ((maskIn[pointNum] == 1.0) and (maskIn[pointNum + lagCad] == 1.0)) {
					printf("y[%d]: %e\n", pointNum + lagCad, yIn[pointNum + lagCad]);
                    printf("mask[%d]: %e\n", pointNum + lagCad, maskIn[pointNum + lagCad]);
					printf("y[%d]: %e\n", pointNum, yIn[pointNum]);
                    printf("mask[%d]: %e\n", pointNum, maskIn[pointNum]);
					printf("sfVals[%d]: %e\n", lagCad, sfVals[lagCad]);
                    printf("sfErrVals[%d]: %e\n", lagCad, sfErrVals[lagCad]);
                    printf("count: %f\n", count);
					}
			#endif
			}
		if (count > 0.0) {
			sfVals[lagCad] = sfVals[lagCad]/count;
			sfErrVals[lagCad] = sqrt(sfErrVals[lagCad])/count;
			}
        #ifdef DEBUG_SF
            printf("\n");
            printf("lagVals: %e\n", lagVals[lagCad]);
			printf("count: %f\n", count);
			printf("sfVals[%d]: %e\n", lagCad, sfVals[lagCad]);
			printf("sfErrVals[%d]: %e\n", lagCad, sfErrVals[lagCad]);
            printf("\n");
		#endif
		}
	return 0;
	}
