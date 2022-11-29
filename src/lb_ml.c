/*
 * First Available Server load balancing algorithm.
 *
 * This file implements an algorithm which emerged during a discussion with
 * Steen Larsen, initially inspired from Anshul Gandhi et.al.'s work now
 * described as "packing" in section 3.5:
 *
 *    http://reports-archive.adm.cs.cmu.edu/anon/2012/CMU-CS-12-109.pdf
 *
 * Copyright 2000-2012 Willy Tarreau <w@1wt.eu>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version
 * 2 of the License, or (at your option) any later version.
 *
 */

#include <stdio.h>

#include <import/eb32tree.h>
#include <haproxy/api.h>
#include <haproxy/backend.h>
#include <haproxy/mllb.h>
#include <haproxy/queue.h>
#include <haproxy/server-t.h>


/* Remove a server from a model. It must have previously been dequeued. This
 * function is meant to be called when a server is going down or has its
 * weight disabled.
 *
 * The server's lock and the lbprm's lock must be held.
 */
static inline void ml_remove_from_tree(struct server *s)
{
	s->lb_mlb = NULL;
  s->lb_mlb_id = 0;
}

static void ml_server_take_conn(struct server *s)
{
	HA_RWLOCK_WRLOCK(LBPRM_LOCK, &s->proxy->lbprm.lock);
	if (s->lb_mlb) {
    ModelLB_take_conn(s->lb_mlb, s, s->lb_mlb_id);
	}
	HA_RWLOCK_WRUNLOCK(LBPRM_LOCK, &s->proxy->lbprm.lock);
}

static void ml_server_drop_conn(struct server *s)
{
	HA_RWLOCK_WRLOCK(LBPRM_LOCK, &s->proxy->lbprm.lock);
	if (s->lb_mlb) {
    ModelLB_drop_conn(s->lb_mlb, s, s->lb_mlb_id);
	}
	HA_RWLOCK_WRUNLOCK(LBPRM_LOCK, &s->proxy->lbprm.lock);
}

/* This function updates the server trees according to server <srv>'s new
 * state. It should be called when server <srv>'s status changes to down.
 * It is not important whether the server was already down or not. It is not
 * important either that the new state is completely down (the caller may not
 * know all the variables of a server's state).
 *
 * The server's lock must be held. The lbprm's lock will be used.
 */
static void ml_set_server_status_down(struct server *srv)
{
	struct proxy *p = srv->proxy;

	if (!srv_lb_status_changed(srv))
		return;

	if (srv_willbe_usable(srv))
		goto out_update_state;

	HA_RWLOCK_WRLOCK(LBPRM_LOCK, &p->lbprm.lock);

	if (!srv_currently_usable(srv))
		/* server was already down */
		goto out_update_backend;

	if (srv->flags & SRV_F_BACKUP) {
		p->lbprm.tot_wbck -= srv->cur_eweight;
		p->srv_bck--;

		if (srv == p->lbprm.fbck) {
			/* we lost the first backup server in a single-backup
			 * configuration, we must search another one.
			 */
			struct server *srv2 = p->lbprm.fbck;
			do {
				srv2 = srv2->next;
			} while (srv2 &&
				 !((srv2->flags & SRV_F_BACKUP) &&
				   srv_willbe_usable(srv2)));
			p->lbprm.fbck = srv2;
		}
	} else {
		p->lbprm.tot_wact -= srv->cur_eweight;
		p->srv_act--;
	}

	ModelLB_server_down(srv->lb_mlb, srv, srv->lb_mlb_id);
	ml_remove_from_tree(srv);

 out_update_backend:
	/* check/update tot_used, tot_weight */
	update_backend_weight(p);
	HA_RWLOCK_WRUNLOCK(LBPRM_LOCK, &p->lbprm.lock);

 out_update_state:
	srv_lb_commit_status(srv);
}

/* This function updates the server trees according to server <srv>'s new
 * state. It should be called when server <srv>'s status changes to up.
 * It is not important whether the server was already down or not. It is not
 * important either that the new state is completely UP (the caller may not
 * know all the variables of a server's state). This function will not change
 * the weight of a server which was already up.
 *
 * The server's lock must be held. The lbprm's lock will be used.
 */
static void ml_set_server_status_up(struct server *srv)
{
	struct proxy *p = srv->proxy;

	if (!srv_lb_status_changed(srv))
		return;

	if (!srv_willbe_usable(srv))
		goto out_update_state;

	HA_RWLOCK_WRLOCK(LBPRM_LOCK, &p->lbprm.lock);

	if (srv_currently_usable(srv))
		/* server was already up */
		goto out_update_backend;

	if (srv->flags & SRV_F_BACKUP) {
		srv->lb_tree = &p->lbprm.fas.bck;
		p->lbprm.tot_wbck += srv->next_eweight;
		p->srv_bck++;

		if (!(p->options & PR_O_USE_ALL_BK)) {
			if (!p->lbprm.fbck) {
				/* there was no backup server anymore */
				p->lbprm.fbck = srv;
			} else {
				/* we may have restored a backup server prior to fbck,
				 * in which case it should replace it.
				 */
				struct server *srv2 = srv;
				do {
					srv2 = srv2->next;
				} while (srv2 && (srv2 != p->lbprm.fbck));
				if (srv2)
					p->lbprm.fbck = srv;
			}
		}
	} else {
		srv->lb_tree = &p->lbprm.fas.act;
		p->lbprm.tot_wact += srv->next_eweight;
		p->srv_act++;
	}

  if (srv->lb_mlb_id == 0) {
	  srv->lb_mlb_id = ModelLB_server_up(srv->lb_mlb, srv);
  }

 out_update_backend:
	/* check/update tot_used, tot_weight */
	update_backend_weight(p);
	HA_RWLOCK_WRUNLOCK(LBPRM_LOCK, &p->lbprm.lock);

 out_update_state:
	srv_lb_commit_status(srv);
}

/* This function must be called after an update to server <srv>'s effective
 * weight. It may be called after a state change too.
 *
 * The server's lock must be held. The lbprm's lock will be used.
 */
static void ml_update_server_weight(struct server *srv)
{
	int old_state, new_state;
	struct proxy *p = srv->proxy;

	if (!srv_lb_status_changed(srv))
		return;

	/* If changing the server's weight changes its state, we simply apply
	 * the procedures we already have for status change. If the state
	 * remains down, the server is not in any tree, so it's as easy as
	 * updating its values. If the state remains up with different weights,
	 * there are some computations to perform to find a new place and
	 * possibly a new tree for this server.
	 */
	 
	old_state = srv_currently_usable(srv);
	new_state = srv_willbe_usable(srv);

	if (!old_state && !new_state) {
		srv_lb_commit_status(srv);
		return;
	}
	else if (!old_state && new_state) {
		ml_set_server_status_up(srv);
		return;
	}
	else if (old_state && !new_state) {
		ml_set_server_status_down(srv);
		return;
	}

	HA_RWLOCK_WRLOCK(LBPRM_LOCK, &p->lbprm.lock);

	// if (srv->lb_tree)
	// 	ml_dequeue_srv(srv);

	// if (srv->flags & SRV_F_BACKUP) {
	// 	p->lbprm.tot_wbck += srv->next_eweight - srv->cur_eweight;
	// 	srv->lb_tree = &p->lbprm.fas.bck;
	// } else {
	// 	p->lbprm.tot_wact += srv->next_eweight - srv->cur_eweight;
	// 	srv->lb_tree = &p->lbprm.fas.act;
	// }

	// ml_queue_srv(srv);

	update_backend_weight(p);
	HA_RWLOCK_WRUNLOCK(LBPRM_LOCK, &p->lbprm.lock);

	srv_lb_commit_status(srv);
}

/* This function is responsible for building the trees in case of fast
 * weighted least-conns. It also sets p->lbprm.wdiv to the eweight to
 * uweight ratio. Both active and backup groups are initialized.
 */
void ml_init_server_tree(struct proxy *p, char param_path[])
{
	struct server *srv;
  // printf("!!! ml_init_server_tree on %s !!!\n", param_path);

	p->lbprm.set_server_status_up   = ml_set_server_status_up;
	p->lbprm.set_server_status_down = ml_set_server_status_down;
	p->lbprm.update_server_eweight  = ml_update_server_weight;
	p->lbprm.server_take_conn = ml_server_take_conn;
	p->lbprm.server_drop_conn = ml_server_drop_conn;

	p->lbprm.wdiv = BE_WEIGHT_SCALE;
	for (srv = p->srv; srv; srv = srv->next) {
		srv->next_eweight = (srv->uweight * p->lbprm.wdiv + p->lbprm.wmult - 1) / p->lbprm.wmult;
		srv_lb_commit_status(srv);
	}

	recount_servers(p);
	update_backend_weight(p);

	p->lbprm.ml.act = ModelLB_new(param_path);
  // printf("Created active ModelLB at %p\n", p->lbprm.ml.act);
	p->lbprm.ml.bck = ModelLB_new(param_path);
  // printf("Created backup ModelLB at %p\n", p->lbprm.ml.bck);

	/* queue active and backup servers in two distinct groups */
	for (srv = p->srv; srv; srv = srv->next) {
		if (!srv_currently_usable(srv))
			continue;
		srv->lb_mlb = (srv->flags & SRV_F_BACKUP) ? p->lbprm.ml.bck : p->lbprm.ml.act;
	  srv->lb_mlb_id = ModelLB_server_up(srv->lb_mlb, srv);
    // printf("Assigned srv %p to %p\n", srv, srv->lb_mlb);
	}
}

/* Return next server in backend <p>. If empty, return NULL. Saturated servers are skipped.
 *
 * The lbprm's lock will be used. The server's lock is not used.
 */
struct server *ml_get_next_server(struct proxy *p, struct server *srvtoavoid)
{
	struct server *srv, *avoided;
	void* mlb;
  int remaining_attempts;  // TODO: configurable?
  // printf("!!! ml_get_next_server !!!\n");

	srv = avoided = NULL;

	HA_RWLOCK_RDLOCK(LBPRM_LOCK, &p->lbprm.lock);
	if (p->srv_act)
		mlb = p->lbprm.ml.act;
	else if (p->lbprm.fbck) {
		srv = p->lbprm.fbck;
		goto out;
	}
	else if (p->srv_bck)
		mlb = p->lbprm.ml.bck;
	else {
		srv = NULL;
		goto out;
	}

  ModelLB_update_state(mlb);
  remaining_attempts = 100;
	while (mlb && remaining_attempts-- > 0) {
    struct server *s = (struct server *) ModelLB_select_server(mlb);
    // printf("ml_get_next_server trying %p\n", s);
		if (!s->maxconn || (!s->queue.length && s->served < srv_dynamic_maxconn(s))) {
			if (s != srvtoavoid) {
				srv = s;
				break;
			}
			avoided = s;
		}
	}

	if (!srv)
		srv = avoided;
  // printf("ml_get_next_server: %p\n", srv);
  out:
	HA_RWLOCK_RDUNLOCK(LBPRM_LOCK, &p->lbprm.lock);
	return srv;
}
