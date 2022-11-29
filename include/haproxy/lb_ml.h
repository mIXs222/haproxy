/*
 * include/haproxy/lb_ml.h
 * Model-based load balancing algorithm.
 */

#ifndef _HAPROXY_LB_ML_H
#define _HAPROXY_LB_ML_H

#include <haproxy/api.h>
#include <haproxy/lb_ml-t.h>
#include <haproxy/proxy-t.h>
#include <haproxy/server-t.h>

struct server *ml_get_next_server(struct proxy *p, struct server *srvtoavoid);
void ml_init_server_tree(struct proxy *p, char param_path[]);

#endif /* _HAPROXY_LB_ML_H */

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 * End:
 */
