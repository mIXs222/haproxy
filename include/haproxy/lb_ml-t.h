/*
 * include/haproxy/lb_ml-t.h
 * Model-based load balancing algorithm.
 */

#ifndef _HAPROXY_LB_ML_T_H
#define _HAPROXY_LB_ML_T_H

#include <import/ebtree-t.h>

struct lb_ml {
	void* act;	/* mllb::ModelLB on the active servers */
	void* bck;	/* mllb::ModelLB on the backup servers */
};

#endif /* _HAPROXY_LB_ML_T_H */
